#!/usr/bin/env python3
from __future__ import annotations
import os
import time
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from sys import platform
from typing import Optional, Union

from picograd.print_utils import *
from picograd.backend.function import *
from picograd.backend.cpu.ops import *
from picograd.util import *
from picograd.backend.device import Devices, Device
from picograd.backend.scheduler import Scheduler
from picograd.backend.renderer.cstyle import CStyleRenderer
from picograd.backend.renderer.cuda_renderer import CUDARenderer
from picograd.backend.renderer.metal_renderer import MetalRenderer
from picograd.backend.linearizer import linearize, build_ast
from picograd.viz import recorder as viz

DEBUG = int(os.getenv("DEBUG", 0))
VERBOSE = int(os.getenv("VERBOSE", 0))
LAZY = int(os.getenv("LAZY", 0))
_DEFAULT_DEVICE = None
_DEFAULT_DEVICE_NAME = None

def default_device():
  global _DEFAULT_DEVICE, _DEFAULT_DEVICE_NAME
  cuda = os.getenv("CUDA", "0") == "1"
  metal = os.getenv("METAL", "0") == "1"
  if cuda and metal: raise ValueError("Only one of CUDA=1 or METAL=1 can be set")
  name = Devices.CUDA if cuda else Devices.METAL if metal else Devices.CPU
  if _DEFAULT_DEVICE is None or _DEFAULT_DEVICE_NAME != name:
    _DEFAULT_DEVICE = Device(name)
    _DEFAULT_DEVICE_NAME = name
  return _DEFAULT_DEVICE

# init c++ library
# if platform == "linux" or platform == "linux2": PICOGRAD_LIB = ctypes.CDLL('./lib/libpicograd.so')  # linux
# elif platform == "darwin":  PICOGRAD_LIB = ctypes.CDLL('./picograd/lib/libpicograd.dylib') # OS X
# elif platform == "win32": PICOGRAD_LIB = ctypes.CDLL('./picograd/lib/libpicograd.dll') # Windows
# else: PICOGRAD_LIB = None


class Tensor:
  def __init__(
    self,
    data: Optional[np.array] = None,  # TODO: make it regular list or tuple and convert to numpy array + .numpy() + numpy to Tensor
    name = "t",
    _prev = set(),
    _prev_forward_args: tuple = (),
    _prev_forward_kwargs: dict = {},
    requires_grad = True,
    device = None,
    device_data: Optional[ctypes.c_void_p] = None,
    shape: Optional[Tuple] = None,
    strides: Optional[Tuple[int]] = None,
    lazy=True if LAZY else False,
    dtype = np.float32,
    _owns_device_data = True,
  ):
    # TODO: auto-detect device based on availability and data type
    if device is None:
      device = default_device()

    data = np.array(data) if data is not None else None
    self.lazy = lazy
    # TODO: detect device availability and fall back to CPU if not available
    self.device = list(_prev)[0].device if len(list(_prev)) > 0 else device
    self._ctx = None  # TODO: use context like pytorch

    self.name = name
    self.verbose = bool(VERBOSE)
    self.requires_grad = requires_grad

    self._shape = shape if data is None else data.shape
    self._data = np.zeros(self._shape, dtype=dtype) if data is None else data
    self._grad = Tensor.zeros(self._shape, dtype=dtype, requires_grad=False, device=device, lazy=lazy) if requires_grad else None

    self.realized = False

    # shapetracker
    # NOTE: tensor[i, j] -> index(i, j) = i * stride[0] + j * stride[1] -> tensor.device_data + index(i, j) * itemsize
    if strides is not None:
      self._strides = tuple(strides)
    else:
      if self._data is not None:
        # numpy strides are bytes -> convert to elements
        self._strides = tuple(int(s // self._data.itemsize) for s in self._data.strides)
      elif self._shape is not None:
        self._strides = default_strides(self._shape)
      else:
        self._strides = None
    
    self._prev = tuple(dict.fromkeys(_prev))  # used to be set, but doesn't preserve order
    self._prev_forward_args = _prev_forward_args
    self._prev_forward_kwargs = _prev_forward_kwargs
    self.prev_op = None
    self.layer = None
    self._backward = lambda: None

    self._device_data = device_data # TODO: if device_data is none, shape is not none and device != CPU, allocate memory of shape on the device
    self._owns_device_data = _owns_device_data
    self._device_grad = None
    if device.name != Devices.CPU: self.to(device)

  def __getstate__(self):
    # snapshot host data only: skips unpicklable lambdas, device managers, and dangling device pointers
    return {
      'data': self.data.copy(),
      'name': self.name,
      'requires_grad': self.requires_grad,
      'device': self.device.name.name,
      'dtype': self.dtype,
    }

  def __setstate__(self, state):
    self.__init__(np.array(state['data'], dtype=state['dtype']), name=state['name'],
                  requires_grad=state['requires_grad'], device=Device(Devices[state['device']]), dtype=state['dtype'])

  @property
  def data(self) -> np.ndarray:
    assert self._data is not None or self.device_data is not None, "Tensor data is not initialized."
    if self.lazy and self.prev_op is not None and not self.realized: self.realize()
    if self.device.name != Devices.CPU and self.device_data is not None: self.device.manager.dev_data_to_host(self, free=False)
    if not isinstance(self._data, np.ndarray): self._data = np.array(self._data)
    return self._data
  
  @data.setter
  def data(self, value):
    initial_shape = self.shape
    self._data = value

    # update shapetracker
    if isinstance(value, np.ndarray):
      self._shape = value.shape
      self._strides = tuple(int(s // value.itemsize) for s in value.strides)

    if self.device.name != Devices.CPU and self.device_data is not None:
      if initial_shape != value.shape: self._shape = value.shape
      _, d_value = self.device.manager.np_to_device(value)
      self.device_data = d_value

  @property
  def grad(self):
    return self._grad

  @grad.setter
  def grad(self, value):
    if value is None:
      self._grad = None
      return
    if not isinstance(value, Tensor):
      value = Tensor(value, requires_grad=False, device=self.device, lazy=self.lazy)
    if value.device.name != self.device.name: value.to(self.device)
    value.requires_grad = False
    self._grad = value

  @property
  def device_data(self): return self._device_data

  @device_data.setter
  def device_data(self, value):
    if self.device.manager and self._owns_device_data and self._device_data is not None: self.device.manager.free_device_tensor(self._device_data)
    self._device_data = value
    self._owns_device_data = value is not None

  @property
  def device_grad(self): return self._grad.device_data if isinstance(self._grad, Tensor) and self._grad.device.name != Devices.CPU else self._device_grad

  @device_grad.setter
  def device_grad(self, value):
    if value is None:
      self._device_grad = None
      return
    if self.device.name != Devices.CPU:
      self._grad = Tensor(device_data=value, shape=self.shape, device=self.device, requires_grad=False, lazy=self.lazy)
    else:
      self._device_grad = value

  @property
  def dtype(self):
    return self._data.dtype if self._data is not None else np.float32

  @dtype.setter
  def dtype(self, value):
    if self._data is not None:
      self._data = self._data.astype(value)

  def item(self, *args): return self.data.item(*args)

  @property
  def shape(self, idxs=None):
    assert self._shape is not None or self._data is not None or self.device_data is not None, "Tensor shape is not initialized."
    if idxs is None: return self._shape if self._data is None else self._data.shape

    if self._data is None:
      if self._shape is not None:
        return self._shape
      else:
        raise ValueError("Tensor shape is not initialized and no device data is available.")

    if idxs is None:
      return self._data.shape
    ret = []
    shp = self._data.shape
    for idx in idxs:
      ret.append(shp[idx])
    
    if len(ret) == 1:
      ret = int(ret[0])
    return ret

  @property
  def strides(self):
    if self._strides is not None: return tuple(self._strides)
    if self._data is not None: return tuple(int(s // self._data.itemsize) for s in self._data.strides)
    return None

  @property
  def is_contiguous(self): return self.strides == default_strides(self.shape)

  @property
  def ndim(self):
    if self._shape is None: return 0
    return len(self._shape)

  @property
  def nbytes(self): return int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)

  def __repr__(self):
    grad_repr = None if self.grad is None else (self.grad.data if self.device.name == Devices.CPU else hex(id(self.grad.device_data)))
    return f"{color_yellow('Tensor')} (name={self.name}, shape={self.shape}, strides={self.strides}, device={self.device.name}, data={self.data if self.device.name == Devices.CPU else hex(id(self.device_data))}{f', grad={grad_repr}' if self.requires_grad else ''}, requires_grad={self.requires_grad}, prev_op={self.prev_op}, prev_tensors={len(self._prev)})"

  def __len__(self):
    return self.shape[0]
    
  def __del__(self):
    if getattr(self, "_device_data", None) is not None and getattr(self, "_owns_device_data", False) and getattr(self, "device", None) is not None and self.device.manager is not None and self.device.name != Devices.CPU:
      try:
        self.device.manager.free_device_tensor(self._device_data)
      except Exception:
        pass
      self._device_data = None
      self._owns_device_data = False
    if getattr(self, "_device_grad", None) is not None and getattr(self, "device", None) is not None and self.device.manager is not None and self.device.name != Devices.CPU:
      try:
        self.device.manager.free_device_tensor(self._device_grad)
      except Exception:
        pass
      self._device_grad = None

  # TODO: implement all tensor generators + cuda
  @staticmethod
  def random(shape: Tuple[int], dtype=np.float32, *args, **kwargs):
    return Tensor(np.random.randn(*shape).astype(dtype), *args, **kwargs)
  
  @staticmethod
  def zeros(shape: Tuple[int], dtype=np.float32, *args, **kwargs):
    return Tensor(np.zeros(shape, dtype=dtype), *args, **kwargs)
  
  @staticmethod
  def ones(shape: Tuple[int], dtype=np.float32, *args, **kwargs):
    return Tensor(np.ones(shape, dtype=dtype), *args, **kwargs)
  
  @staticmethod
  def eye(n: int, dtype=np.float32, *args, **kwargs):
    return Tensor(np.eye(n, dtype=dtype), *args, **kwargs)

  def to(self, device: Device):
    """Tranfers tensor to the specified device."""

    if device.name == Devices.CPU:
      if self.device_data is not None: self.device.manager.tensor_to_host(self)
      self.device = device
      return self

    self.device = device
    if device.name != Devices.CPU :
      self._data = self._data.astype(np.float32)
      if self._grad is not None:
        self._grad.dtype = np.float32
        if self._grad.device.name != device.name: self._grad.to(device)
      self.device.manager.tensor_to_device(self)

    return self

  def backward(self):
    topo = []
    visited = set()
    stack = [self]
    while stack:
      v = stack.pop()
      if v not in visited:
        visited.add(v)
        stack.append(v)
        if not isinstance(v, Tensor): continue
        for child in v._prev:
          if child not in visited: stack.append(child)
      elif v not in topo: topo.append(v)

    for node in topo:
      if isinstance(node, Tensor) and node.lazy and node.prev_op is not None and not node.realized:
        node.realize()

    self.grad = Tensor.ones(self.shape, requires_grad=False, device=self.device, lazy=self.lazy)
    for node in reversed(topo):
      if isinstance(node, Tensor):
        node._backward()

    for node in topo:
      if isinstance(node, Tensor) and node.requires_grad and isinstance(node.grad, Tensor) and node.grad.lazy and node.grad.prev_op is not None and not node.grad.realized:
        node.grad.realize()

    for node in topo:
      if isinstance(node, Tensor) and node is not self and node.prev_op is not None:
        node.clear_graph()
      if isinstance(node, Tensor) and isinstance(node.grad, Tensor):
        node.grad.clear_graph()

  def clear_graph(self):
    self._prev = ()
    self._prev_forward_args = ()
    self._prev_forward_kwargs = {}
    self._backward = lambda: None
    return self

  # pretty print the graph for this tensor backwards
  def print_graph(self, verbose=False):
    tmp = list(reversed(list(self._prev.copy())))
    tmp.insert(0, self)

    topo = []
    visited = set()
    stack = [self]
    while stack:
      v = stack.pop()
      if v not in visited:
        visited.add(v)
        stack.append(v)
        for child in v._prev:
          if child not in visited: stack.append(child)
      elif v not in topo: topo.append(v)
    for node in reversed(topo):
      print(node)
      if verbose:
        print("[data]\n", node.data)
        print("[grad]\n", node.grad)
      if node.prev_op != None:
        print("====++++****++++====\n[OP]:", node.prev_op ,"\n====++++****++++====")

  # TODO: handle cuda as well
  def float(self):
    self.data = self.data.astype(np.float32)
    self.dtype = np.float32
    return self

  def long(self):
    self.data = self.data.astype(np.int64)
    self.dtype = np.int64
    return self

  def scalar_to_tensor(self, value: Union[int, float]):
    assert isinstance(value, Tensor) or isinstance(value, (int, float, np.ndarray)), f"Operand {value} must be a Tensor, ndarray, or scalar (int/float). Got {type(value)}."
    return Tensor(value if isinstance(value, np.ndarray) else [value], name=str(value), requires_grad=False, device=self.device, lazy=self.lazy) if isinstance(value, (int, float, np.ndarray)) else value

  def get_renderer(self) -> CStyleRenderer:
    if self.device.name == Devices.CPU: raise NotImplementedError("CPU renderer not implemented yet")
    if self.device.name == Devices.CUDA: return CUDARenderer(arch="sm_80")
    if self.device.name == Devices.METAL: return MetalRenderer()
    raise NotImplementedError(f"Renderer not implemented for {self.device.name} yet")

  def detach(self):
    return Tensor(
      data=self.data.copy() if self.device.name == Devices.CPU else None,
      device_data=self.device_data,
      shape=self.shape,
      strides=self.strides,
      requires_grad=False,
      device=self.device,
      lazy=self.lazy,
      _owns_device_data=False,
    )

  def _accumulate_grad(self, grad: "Tensor"):
    if not self.requires_grad: return
    grad.requires_grad = False
    self._grad = grad if self._grad is None else self._grad + grad

  def _sum_to_shape(self, grad: "Tensor", shape: Tuple[int, ...]):
    while len(grad.shape) > len(shape): grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
      if dim == 1 and grad.shape[axis] != 1: grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape) if grad.shape != shape else grad

  def _backward_from_op(self, out: "Tensor", op_name: OPS, inputs: Tuple["Tensor", ...], forward_args: Tuple, forward_kwargs: dict):
    grad_out = out.grad
    if grad_out is None: return
    a = inputs[0]
    operands = inputs[1:]

    if op_name == OPS.ADD:
      b = operands[0]
      a._accumulate_grad(self._sum_to_shape(grad_out, a.shape))
      b._accumulate_grad(self._sum_to_shape(grad_out, b.shape))
      return

    if op_name == OPS.MUL:
      b = operands[0]
      a._accumulate_grad(self._sum_to_shape(b.detach() * grad_out, a.shape))
      b._accumulate_grad(self._sum_to_shape(a.detach() * grad_out, b.shape))
      return

    if op_name == OPS.DOT:
      b = operands[0]
      if a.requires_grad: a._accumulate_grad(grad_out.dot(b.detach().T))
      if b.requires_grad: b._accumulate_grad(a.detach().T.dot(grad_out))
      return

    if op_name in (OPS.Reshape, OPS.View, OPS.Flatten):
      a._accumulate_grad(grad_out.reshape(a.shape))
      return

    if op_name == OPS.Unsqueeze:
      axis = forward_args[0]
      if axis < 0: axis += len(a.shape) + 1
      a._accumulate_grad(grad_out.squeeze(axis))
      return

    if op_name == OPS.Squeeze:
      axis = forward_args[0]
      a._accumulate_grad(grad_out.unsqueeze(axis).reshape(a.shape))
      return

    if op_name == OPS.Transpose:
      axes = forward_args[0] if len(forward_args) > 0 else tuple(reversed(range(len(a.shape))))
      a._accumulate_grad(grad_out.permute(*tuple(np.argsort(axes))))
      return

    if op_name == OPS.Permute:
      axes = forward_args[0]
      a._accumulate_grad(grad_out.permute(*tuple(np.argsort(axes))))
      return

    if op_name == OPS.Expand:
      a._accumulate_grad(self._sum_to_shape(grad_out, a.shape))
      return

    if op_name == OPS.SUM:
      axis, keepdims = forward_args
      g = grad_out
      if axis is not None and not keepdims: g = g.unsqueeze(axis)
      a._accumulate_grad(g.expand(*a.shape))
      return

    if op_name == OPS.ReLU:
      mask = Tensor((a.detach().data > 0).astype(np.float32), requires_grad=False, device=a.device, lazy=a.lazy)
      a._accumulate_grad(grad_out * mask)
      return

    if op_name == OPS.Tanh and a.device.name == Devices.CPU:
      a._accumulate_grad(Tensor((1 - np.tanh(a.detach().data)**2) * grad_out.data, requires_grad=False, device=a.device, lazy=a.lazy))
      return

    if op_name == OPS.Sigmoid and a.device.name == Devices.CPU:
      sig = 1 / (1 + np.exp(-a.detach().data))
      a._accumulate_grad(Tensor(sig * (1 - sig) * grad_out.data, requires_grad=False, device=a.device, lazy=a.lazy))
      return

    if op_name == OPS.Softmax:
      axis = forward_args[0] if len(forward_args) > 0 and forward_args[0] is not None else -1
      data = out.detach()
      dot = (grad_out * data).sum(axis=axis, keepdims=True)
      a._accumulate_grad((grad_out + (dot * Tensor(np.full(dot.shape, -1, dtype=np.float32), requires_grad=False, device=a.device, lazy=a.lazy))) * data)
      return

    if op_name == OPS.MEAN and a.device.name == Devices.CPU:
      axis, keepdims = forward_args
      g = grad_out.data
      if not keepdims and axis is not None: g = np.expand_dims(g, axis=axis)
      a._accumulate_grad(Tensor(np.ones_like(a.data) * g / a.data.size, requires_grad=False, device=a.device, lazy=a.lazy))
      return

    if op_name == OPS.MAX and a.device.name == Devices.CPU:
      axis, keepdims = forward_args
      g = grad_out.data
      if not keepdims and axis is not None: g = np.expand_dims(g, axis=axis)
      a._accumulate_grad(Tensor((a.data == np.max(a.data, axis=axis, keepdims=True)) * g, requires_grad=False, device=a.device, lazy=a.lazy))
      return

    if op_name == OPS.MIN and a.device.name == Devices.CPU:
      axis, keepdims = forward_args
      g = grad_out.data
      if not keepdims and axis is not None: g = np.expand_dims(g, axis=axis)
      a._accumulate_grad(Tensor((a.data == np.min(a.data, axis=axis, keepdims=True)) * g, requires_grad=False, device=a.device, lazy=a.lazy))
      return

    if op_name == OPS.STD and a.device.name == Devices.CPU:
      axis, keepdims = forward_args
      g = grad_out.data
      mean = np.mean(a.data, axis=axis, keepdims=True)
      std = np.std(a.data, axis=axis, keepdims=True)
      if not keepdims and axis is not None: g = np.expand_dims(g, axis=axis)
      a._accumulate_grad(Tensor((a.data - mean) * g / (std * a.data.size), requires_grad=False, device=a.device, lazy=a.lazy))
      return

    return

  def numpy(self):
    if self.lazy: self.realize()
    return self.data

  def tolist(self):
    if self.lazy: self.realize()
    return self.data.tolist()

  def realize(self):
    if not self.lazy:
      if self.device.name != Devices.CPU and self.device_data is not None: self.device.manager.tensor_to_host(self)
      return

    # Leaf tensors hold data directly (params, inputs) — nothing to compile/schedule
    if self.prev_op is None or len(self._prev) == 0:
      if self.device.name != Devices.CPU and self.device_data is not None:
        self.device.manager.dev_data_to_host(self, free=False)  # sync device->host, keep device copy
      self.realized = True
      return

    if viz.enabled():
      realize_start = time.perf_counter()
      viz.record("realize_start", tensor=self)

    renderer = self.get_renderer()
    ast = linearize(build_ast(self))
    scheduler = Scheduler(ast, renderer)
    schedule = scheduler.create_schedule()

    if viz.enabled(): viz.record("schedule", tensor=self, ast_nodes=len(ast), schedule_items=len(schedule), ops=[node.op for node in ast])
    try:
      scheduler.run_schedule()
      for node in ast:
        if node.tensor.prev_op is not None: node.tensor.realized = True
    finally:
      if viz.enabled(): viz.record("realize_end", tensor=self, ast_nodes=len(ast), schedule_items=len(schedule), duration_ms=(time.perf_counter() - realize_start) * 1000.0)

  def from_op(
    self,
    op_name: str,
    shape: Optional[Tuple] = None,
    strides: Optional[Tuple[int]] = None,
    operands: Tuple["Tensor"] = (),
    forward_args: Tuple = (),
    forward_kwargs: dict = {},
  ) -> Tensor:
    """
    Generalized op creation for tensor operations.
    
    Args:
        op_name: Name of the operation (e.g., OPS.ADD, OPS.DOT, etc).
        operands: Other tensor arguments involved in the op.
        forward_args: Extra positional arguments for func.forward (e.g., stride, padding).
        forward_kwargs: Extra keyword arguments for func.forward.
    Returns: New tensor resulting from the operation.
    """

    operands = tuple([self.scalar_to_tensor(t) for t in operands])
    tensor_inputs = (self,) + operands
    prev = tensor_inputs
    requires_grad = any(t.requires_grad for t in tensor_inputs)
    
    shared_device_data = op_name in MOVEMENT_OPS or op_name == OPS.Transpose
    out_data = (self.data if self.device.name == Devices.CPU else self.device_data) if shared_device_data else None
    owns_device_data = self.device.name != Devices.CPU and not shared_device_data
    if not self.lazy:
      func = get_op(op_name, self.device.name)
      out_data = func.forward(*tensor_inputs, *forward_args, **forward_kwargs)
      owns_device_data = self.device.name != Devices.CPU and not shared_device_data

    if self.device.name == Devices.CPU:
      out = Tensor(
        out_data,
        shape=shape if shape is not None else self.shape,
        strides=strides if strides is not None else default_strides(shape if shape is not None else self.shape),
        _prev=prev,
        _prev_forward_args=forward_args,
        _prev_forward_kwargs=forward_kwargs,
        device=self.device,
        requires_grad=requires_grad,
        lazy=self.lazy,
        _owns_device_data=owns_device_data,
      )
    else:
      out = Tensor(
        device_data=out_data,
        shape=shape if shape is not None else self.shape,
        strides=strides if strides is not None else default_strides(shape if shape is not None else self.shape),
        _prev=prev,
        _prev_forward_args=forward_args,
        _prev_forward_kwargs=forward_kwargs,
        device=self.device,
        requires_grad=requires_grad,
        lazy=self.lazy,
        _owns_device_data=owns_device_data,
      )

    out.prev_op = op_name
    out._backward = lambda: self._backward_from_op(out, op_name, tensor_inputs, forward_args, forward_kwargs)
    viz.record("op_created", op=op_name, tensor=out, inputs=tensor_inputs, shape=out.shape, strides=out.strides, forward_args=forward_args, forward_kwargs=forward_kwargs)
    return out

  def shape_for_reduce(self, axis, keepdims):
    # full reduction (sum all elements) - result is scalar or 1-element tensor
    if axis is None: return (1,)

    # normalize negative axis
    if axis < 0: axis += len(self.shape)
    assert 0 <= axis < len(self.shape), "axis out of bounds"

    return tuple(1 if i == axis else s for i, s in enumerate(self.shape)) if keepdims else tuple(s for i, s in enumerate(self.shape) if i != axis)

  def strides_for_reduce(self, axis, keepdims, out_shape):
      if axis is None: return (0,) if keepdims else ()
      return default_strides(out_shape)

  def contiguous(self):
    if self.is_contiguous: return self
    # allocate contiguous buffer and copy using a strided->contiguous copy kernel (similar to add_strided logic but copying)
    if self.device.name == Devices.CPU:
      new = Tensor(shape=self.shape, device=self.device, requires_grad=self.requires_grad)
      new._data = np.empty(self.shape, dtype=self.dtype)
      # use numpy fancy indexing or np.copyto with view:
      np.copyto(new._data, self.data)  # copying from strided view into contiguous array
      new._strides = default_strides(new._shape)
      new.is_contiguous = True
      return new

    # TODO: test against other movement ops
    # TODO: !not memory/speed efficient! - requires kernel
    # copy non-strided data to host, make contiguous on host, copy back to device
    numel = required_numel(self.shape, self.strides)
    host_flat = np.zeros(numel, dtype=self.dtype)
    self.device.manager.copy_data_to_host(self.device_data, host_flat)
    # print(hex(self.device_data.value), host_flat)
    strided_temp = as_strided(host_flat, shape=(self.shape), strides=(tuple(s * self.dtype.itemsize for s in self.strides)))
    contig_host = np.ascontiguousarray(strided_temp)
    dst_ptr = self.device.manager.allocate_device_memory(contig_host)
    self.device.manager.copy_data_to_device(dst_ptr, contig_host)
    self.device_data = dst_ptr
    self._owns_device_data = True
    self._strides = default_strides(self.shape)
    return self

  @staticmethod
  def cat(tensors: list["Tensor"], axis=0):
    """Concatenate a sequence of tensors along an existing axis"""

    assert len(tensors) > 0, "Cat expects a non-empty list of tensors"
    # assert all(isinstance(t, "Tensor") for t in tensors), "Cat expects a list of Tensors"
    base_shape = list(tensors[0].shape)
    for t in tensors[1:]:
      for i, (a, b) in enumerate(zip(base_shape, t.shape)):
        if i == axis:
          continue
        assert a == b, f"Cat expects all tensors to have the same shape except along axis {axis}, but got shapes {base_shape} vs {t.shape}"

    device = tensors[0].device
    requires_grad = any(t.requires_grad for t in tensors)

    datas = [t.data for t in tensors]
    out_data = np.concatenate(datas, axis=axis)
    out = Tensor(
      out_data,
      _prev=tuple(dict.fromkeys(tensors)),
      device=device,
      requires_grad=requires_grad
    )
    out.prev_op = OPS.Cat

    def _backward():
      if out.grad is None: return
      sizes = [t.shape[axis] for t in tensors]
      grads = np.split(out.grad, np.cumsum(sizes)[:-1], axis=axis)
      for t, g in zip(tensors, grads):
        if not t.requires_grad: continue
        if t.grad is None:
          t.grad = g
        else:
          t.grad += g
    out._backward = _backward
    return out

  @staticmethod
  def stack(tensors: list["Tensor"], axis=0):
    """
    Stack tensors along a new axis (wrapper over cat)
    Equivalent to: cat([t.unsqueeze(axis) for t in tensors], axis=axis)
    """
    assert len(tensors) > 0, "Stack expects a non-empty list of tensors"
    return Tensor.cat([t.unsqueeze(axis) for t in tensors], axis=axis)

  def masked_fill(self, mask, value):
    self.data = np.where(mask.data if isinstance(mask, Tensor) else mask, value, self.data)
    return self

  # Slices
  # def __getitem__(self, indices):         return self.data[indices]
  # def __setitem__(self, indices, value):  self.data[indices] = value
  def __getitem__(self, indices):         return Tensor(self.data[indices]) # FIXME: tempfix
  def __setitem__(self, indices, value):  self.data[indices] = value.data if isinstance(value, Tensor) else value
  def __equal__(self, other):             return np.equal(self.data, other.data)
  # TODO: overload comparisons

  # Movement Ops
  # TODO: only reshape and flatten create new tensor, others just change shape/strides, but they try view first (+ view if contiguous)
  # FIXME: view should not create a new tensor, but just change the shape and stride of the current one while reshape keeps the memory layout contiguous for the new tensor
  def reshape(self, *args, **kwargs): shape = args if len(args) > 1 else args[0]; return self.from_op(OPS.Reshape, forward_args=(shape,), forward_kwargs=kwargs, shape=shape)
  def view(self, *shape):             return self.from_op(OPS.Reshape, forward_args=(shape,), shape=((shape[0],) if len(shape) == 1 else shape))
  def flatten(self):                  return self.reshape(-1) # TODO: axis
  def unsqueeze(self, axis):
    ndim = self.ndim
    if axis < 0: axis += ndim + 1
    assert 0 <= axis <= ndim, f"unsqueeze axis {axis} out of range for ndim={ndim}"
    shape = tuple(self.shape[:axis]) + (1,) + tuple(self.shape[axis:])
    strides = tuple(self.strides[:axis]) + (0,) + tuple(self.strides[axis:])
    return self.from_op(OPS.Unsqueeze, forward_args=(axis,), shape=shape,strides=strides)

  def squeeze(self, axis=0):
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(a + self.ndim if a < 0 else a for a in axes)
    shape = tuple(s for i, s in enumerate(self.shape) if i not in axes)
    strides = tuple(s for i, s in enumerate(self.strides) if i not in axes)
    return self.from_op(OPS.Squeeze, forward_args=(axis,), shape=shape, strides=strides)
  def expand(self, *sizes):           return self.from_op(OPS.Expand, forward_args=(sizes,), shape=sizes if len(sizes) > 1 else sizes[0], strides=tuple(s if o == n else 0 for o, n, s in zip(self.shape, sizes, self.strides)))
  def permute(self, *axes):           return self.from_op(OPS.Permute, forward_args=(axes,), shape=tuple(self.shape[i] for i in axes), strides=tuple(self.strides[i] for i in axes))
  @property
  def T(self):                        return self.from_op(OPS.Transpose, shape=(tuple(reversed(self.shape))), strides=tuple(reversed(self.strides)))
  def transpose(self, dim0: int, dim1: int):
    ndim = len(self.shape)
    if dim0 < 0: dim0 += ndim
    if dim1 < 0: dim1 += ndim
    axes = list(range(ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    axes = tuple(axes)
    return self.from_op(OPS.Transpose, forward_args=(axes,), shape=tuple(self.shape[i] for i in axes))


  # Binary Ops
  def __add__(self, other):           return self.from_op(OPS.ADD, operands=(other,))
  def __mul__(self, other):           return self.from_op(OPS.MUL, operands=(other,))
  def __matmul__(self, other):        return self.dot(other)
  # NOTE: if lazy, this uses the matmul trick [ https://mesozoic-egg.github.io/tinygrad-notes/20241203_matmul.html ], results in non-contiguous elementwise
  def dot(self, other):
    if not self.lazy: return self.from_op(OPS.DOT, operands=(other,),shape=self.shape[:-1] + (other.shape[-1],))

    assert self.ndim >= 2 and other.ndim >= 2, "dot expects ndim >= 2"
    assert self.shape[-1] == other.shape[-2], f"Cannot matmul {self.shape} and {other.shape}"

    M, K = self.shape[-2:]
    N = other.shape[-1]
    batch = broadcast_shape(self.shape[:-2], other.shape[:-2])

    a = self
    b = other
    while a.ndim - 2 < len(batch): a = a.unsqueeze(0)
    while b.ndim - 2 < len(batch): b = b.unsqueeze(0)
    a = a.expand(*batch, M, K).unsqueeze(-1).expand(*batch, M, K, N)
    b = b.expand(*batch, K, N).unsqueeze(-3).expand(*batch, M, K, N)
    return (a * b).sum(axis=-2)
      
  def __pow__(self, other):           return self.from_op(OPS.POW, operands=(other,))
  def __radd__(self, other):          return self + other
  def __sub__(self, other):           return self + (-other)
  def __rsub__(self, other):          return other + (-self)
  def __rmul__(self, other):          return self * other
  def __truediv__(self, other):       return self * other**-1
  def __rtruediv__(self, other):      return other * self**-1

  def linear(self, weight: "Tensor", bias: Optional["Tensor"] = None):
    x = self * weight if len(weight.shape) == 1 else self.dot(weight)
    if bias is not None and self.lazy and bias.shape != x.shape:
      bias = bias.reshape(*((1,) * (len(x.shape) - len(bias.shape))), *bias.shape).expand(*x.shape)
    return x + bias if bias is not None else x

  # TODO: double check this
  # TODO: Instead of _pool manually constructing a Tensor, make it behave like your other movement ops.
  def _pool(self, k: Tuple[int, int], stride: int, dilation: int):
    """
    Input: (B, C, H, W)
    Output: (B, C, out_h, out_w, k_h, k_w)
    """

    assert len(k) == 2, "_pool currently expects a 2D kernel"
    k_h, k_w = k
    h, w = self.shape[-2:]

    # Effective kernel size when dilation > 1
    eff_k_h = dilation * (k_h - 1) + 1
    eff_k_w = dilation * (k_w - 1) + 1

    out_h = (h - eff_k_h) // stride + 1
    out_w = (w - eff_k_w) // stride + 1

    assert out_h > 0 and out_w > 0, (
      f"Kernel {k} with dilation={dilation} is too large "
      f"for input spatial shape {(h, w)}"
    )

    shape = self.shape[:-2] + (out_h, out_w, k_h, k_w)
    *leading_strides, h_stride, w_stride = self.strides # B, C, H, W
    strides = tuple(leading_strides) + (
      h_stride * stride,    # move output window vertically
      w_stride * stride,    # move output window horizontally
      h_stride * dilation,  # move inside kernel vertically
      w_stride * dilation,  # move inside kernel horizontally
    )

    if self.device.name == Devices.CPU:
      view = as_strided(
        self.data,
        shape=shape,
        strides=tuple(s * self.data.itemsize for s in strides),
      )
      return Tensor(
        view,
        _prev=(self,),
        device=self.device,
        requires_grad=self.requires_grad,
        lazy=self.lazy
      )

    return Tensor(
      device_data=self.device_data,
      shape=shape,
      strides=strides,
      _prev=(self,),
      device=self.device,
      requires_grad=self.requires_grad,
      lazy=self.lazy,
      _owns_device_data=False,
    )

  # TODO: to support padding > 0, we need a more advanced ShapeTracker that supports an index-validity mask, closer to tinygrad
  def conv2d(self, weight: "Tensor", in_channels: int, out_channels: int, stride: int = 1, padding: int = 0, bias: "Tensor" = None):
    if self.lazy:
      assert padding == 0, "lazy conv2d padding not implemented yet"

      x = self._pool(k=(weight.shape[-2:]), stride=stride, dilation=1)  # TODO: dilation could be an arg (?)
      B, IC, OH, OW, KH, KW = x.shape
      OC = weight.shape[0]

      assert IC == in_channels, f"Input channels {IC} does not match weight in_channels {in_channels}"
      assert KH == weight.shape[-2] and KW == weight.shape[-1], f"Weight kernel size {weight.shape[-2:]} does not match input kernel size {(KH, KW)}"
      assert OC == out_channels, f"Weight out_channels {OC} does not match specified out_channels {out_channels}"

      x = x.unsqueeze(1)  # (B, IC, OH, OW, KH, KW) -> (B, 1, IC, OH, OW, KH, KW)
      x = x.expand(B, OC, IC, OH, OW, KH, KW)     # (B, 1, IC, OH, OW, KH, KW) -> (B, OC, IC, OH, OW, KH, KW)
      w = weight.reshape(1, OC, IC, 1, 1, KH, KW) # (OC, IC, KH, KW) -> (1, OC, IC, 1, 1, KH, KW)
      w = w.expand(B, OC, IC, OH, OW, KH, KW)     # (1, OC, IC, 1, 1, KH, KW) -> (B, OC, IC, OH, OW, KH, KW)

      out = x * w
      out = out.sum(axis=-1)  # (B, OC, IC, OH, OW, KH, KW) -> (B, OC, IC, OH, OW, KH)
      out = out.sum(axis=-1)  # -> (B, OC, IC, OH, OW)
      out = out.sum(axis=2)   # -> (B, OC, OH, OW)

      if bias is not None: out = out + bias.reshape(1, OC, 1, 1).expand(B, OC, OH, OW)

      return out

    if bias is None:
      bias = Tensor(np.zeros((out_channels,), dtype=np.float32), name="bias", requires_grad=False, device=self.device)
    return self.from_op(
      OPS.Conv2D,
      shape=(
        self.shape[0],
        out_channels,
        (self.shape[2] - weight.shape[2] + 2 * padding) // stride + 1,
        (self.shape[3] - weight.shape[3] + 2 * padding) // stride + 1
      ),
      operands=(weight, bias),
      forward_args=(in_channels, out_channels, stride, padding),
    )

  # Unary Ops
  # TODO: support add, mul with scalars + fix and test these ops
  def __neg__(self):            return self * Tensor([-1], name="-1", requires_grad=False, device=self.device)
  def sqrt(self):               return self ** 0.5
  def relu(self):               return self.from_op(OPS.ReLU,)
  def softmax(self, axis=None): return self.from_op(OPS.Softmax, forward_args=(axis,))
  def tanh(self):               return self.from_op(OPS.Tanh)
  def sigmoid(self):            return self.from_op(OPS.Sigmoid)

  # Reduce Ops
  # TODO: fix other reduce ops as well
  def sum(self, axis=None, keepdims=False):     return self.from_op(OPS.SUM, forward_args=(axis, keepdims), shape=(out_shape := self.shape_for_reduce(axis, keepdims)), strides=self.strides_for_reduce(axis, keepdims, out_shape))
  def mean(self,axis=None, keepdims=False):
    if self.lazy:
      axes = tuple(range(self.ndim)) if axis is None else (axis,) if isinstance(axis, int) else tuple(axis)
      axes = tuple(a + self.ndim if a < 0 else a for a in axes)
      denom = 1
      for a in axes: denom *= self.shape[a]
      out = self
      for a in sorted(axes, reverse=True): out = out.sum(axis=a, keepdims=keepdims)
      scale = Tensor(np.full(out.shape, 1.0 / denom, dtype=np.float32), requires_grad=False, device=self.device, lazy=self.lazy)
      return out * scale
    return self.from_op(OPS.MEAN, forward_args=(axis, keepdims))
  def max(self, axis=None, keepdims=False):     return self.from_op(OPS.MAX, forward_args=(axis, keepdims))
  def min(self, axis=None, keepdims=False):     return self.from_op(OPS.MIN, forward_args=(axis, keepdims))
  def std(self, axis=None, keepdims=False):     return self.from_op(OPS.STD, forward_args=(axis, keepdims))
  def argmax(self, axis=None, keepdims=False):  return self.from_op(OPS.ARGMAX, forward_args=(axis, keepdims))
  def argmin(self, axis=None, keepdims=False):  return self.from_op(OPS.ARGMIN, forward_args=(axis, keepdims))
  def maxpool2d(self, filter=(2,2), stride=1):  return self.from_op(OPS.MaxPool2D, forward_args=(filter, stride))
  def avgpool2d(self, filter=(2,2), stride=1):  return self.from_op(OPS.AvgPool2D, forward_args=(filter, stride))
