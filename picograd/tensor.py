#!/usr/bin/env python3
import os
import ctypes
import numpy as np
from sys import platform
from typing import Optional

from picograd.print_utils import *
from picograd.backend.function import *
from picograd.backend.cpu.ops import *
from picograd.util import *
from picograd.backend.device import Devices, Device

VERBOSE = int(os.getenv("VERBOSE", 0))

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
    requires_grad = True,
    device = Device(Devices.CPU),
    device_data: Optional[ctypes.c_void_p] = None,
    shape: Optional[Tuple] = None,
  ):

    data = np.array(data) if data is not None else None
    self.device= list(_prev)[0].device if len(list(_prev)) > 0 else device
    self._ctx = None  # TODO: use context like pytorch

    self.name = name
    self._shape = shape if data is None else data.shape
    self.debug = DEBUG
    self.verbose = bool(VERBOSE)
    self.requires_grad = requires_grad

    self._data = data if data is not None else np.zeros(self._shape)
    self._grad = np.zeros(self._shape) if requires_grad else None

    self._prev = set(_prev)
    self.prev_op = None
    self.layer = None
    self._backward = lambda: None

    self._device_data = device_data
    self._device_grad = None
    if device.name != Devices.CPU: self.to(device)

  @property
  def data(self) -> np.ndarray:
    assert self._data is not None or self.device_data is not None, "Tensor data is not initialized."

    if self.device.name != Devices.CPU and self.device_data is not None:
      self.device.manager.dev_data_to_host(self, free=False)

    if not isinstance(self._data, np.ndarray):
      self._data = np.array(self._data)
    return self._data
  
  @data.setter
  def data(self, value):
    initial_shape = self.shape
    self._data = value
    if self.device.name != Devices.CPU and self.device_data is not None:
      if initial_shape != value.shape:
        self._shape = value.shape
      self.device.manager.free_device_tensor(self.device_data)
      _, self._device_data = self.device.manager.np_to_device(value)

  @property
  def grad(self):
    assert self._grad is not None or self.device_grad is not None, "Tensor grad is not initialized."

    if self.device.name != Devices.CPU and self.device_grad is not None:
      self.device.manager.dev_grad_to_host(self, free=False)
    return self._grad

  @grad.setter
  def grad(self, value):
    initial_shape = self.shape
    self._grad = value
    # assert initial_shape == value.shape, "Tensor data shape does not match the initialized shape."  # TODO: check if we need grad.shape == data.shape
    if self.device.name != Devices.CPU and self.device_grad is not None:
      self.device.manager.free_device_tensor(self.device_grad)
      _, self._device_grad = self.device.manager.np_to_device(value)

  @property
  def device_data(self): return self._device_data

  @device_data.setter
  def device_data(self, value):
    if self.device.manager: self.device.manager.free_device_tensor(self._device_data)
    self._device_data = value

  @property
  def device_grad(self): return self._device_grad

  @device_grad.setter
  def device_grad(self, value):
    if self.device.manager: self.device.manager.free_device_tensor(self._device_grad)
    self._device_grad  = value

  @property
  def dtype(self):
    return self._data.dtype if self._data is not None else np.float32

  @dtype.setter
  def dtype(self, value):
    if self._data is not None:
      self._data = self._data.astype(value)

  @property
  def T(self): return Tensor(self.data.T, _prev=(self,), device=self.device)

  @property
  def item(self, *args): return self.data.item(*args)

  @property
  def shape(self, idxs=None):
    assert self._data is not None or self.device_data is not None, "Tensor shape is not initialized."
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

  def __repr__(self):
    if self.verbose:
      return f"{color_yellow('Tensor')} (name={self.name}, shape={self.shape}, device={self.device.name}, data=\n{self.data}\n, grad=\n{self.grad}, prev_op={self.prev_op}, prev_tensors={len(self._prev)})"
    else:
      return f"{color_yellow('Tensor')} (name={self.name}, shape={self.shape}, device={self.device.name}, data=\n{self.data}\n, prev_op={self.prev_op}, prev_tensors={len(self._prev)})"
    
  def __del__(self):
    # if self.device_data is not None:
    #   free_device_tensor(self.device.manager, self.device_data)
    #   self.device_data = None

    # if self.device_grad is not None:
    #   free_device_tensor(self.device.manager, self.device_grad)
    #   self.device_grad = None
    pass

  # TODO: implement all tensor generators + cuda
  @staticmethod
  def random(shape: Tuple[int], name="t", dtype=np.float32, *args, **kwargs):
    return Tensor(np.random.randn(*shape).astype(dtype), *args, **kwargs)
  
  @staticmethod
  def zeros(shape: Tuple[int], name="t", dtype=np.float32, *args, **kwargs):
    return Tensor(np.zeros(shape, dtype=dtype), *args, **kwargs)
  
  @staticmethod
  def ones(shape: Tuple[int], name="t", dtype=np.float32, *args, **kwargs):
    return Tensor(np.ones(shape, dtype=dtype), *args, **kwargs)
  
  @staticmethod
  def eye(n: int, name="t", dtype=np.float32, *args, **kwargs):
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
      if self._grad is not None: self._grad = self._grad.astype(np.float32)
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
    for node in reversed(topo):
      if isinstance(node, Tensor):
        node._backward()

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

  def create_op(
      self,
      op_name: str,
      shape: Optional[Tuple] = None,
      operands: Tuple["Tensor"] = (),
      forward_args: Tuple = (),
      forward_kwargs: dict = {},
    ):
    """
    Generalized op creation for tensor operations.
    
    Args:
        op_name: Name of the operation (e.g., OPS.ADD, OPS.DOT, etc).
        operands: Other tensor arguments involved in the op.
        forward_args: Extra positional arguments for func.forward (e.g., stride, padding).
        forward_kwargs: Extra keyword arguments for func.forward.
    Returns: New tensor resulting from the operation.
    """

    func = get_op(op_name, self.device.name)
    tensor_inputs = (self,) + operands
    prev = tensor_inputs
    out_data = func.forward(*tensor_inputs, *forward_args, **forward_kwargs)

    if self.device.name == Devices.CPU:
      out = Tensor(out_data, _prev=prev, device=self.device)
    else:
      out = Tensor(
        device_data=out_data,
        shape=shape if shape is not None else (self.shape,),
        _prev=prev,
        device=self.device
      )

    out.prev_op = op_name
    out._backward = lambda: func.backward(out.grad if self.device.name == Devices.CPU else out.device_grad)
    return out

  def __getitem__(self, indices):         return self.data[indices]
  def __setitem__(self, indices, value):  self.data[indices] = value
  def __equal__(self, other):             return np.equal(self.data, other.data)

  # Movement Ops
  # FIXME: move to ops (cpu + cuda)
  def reshape(self, *args, **kwargs):
    out = Tensor(self.data.reshape(*args, **kwargs), _prev=(self,))
    original_shape = self.shape
    out.prev_op = OPS.Reshape

    def _backward():
      self.grad += out.grad.reshape(original_shape)
    out._backward = _backward
    return out

  def flatten(self):
    out = Tensor(self.data.flatten(), _prev=(self,), name="flattenout")
    original_shape = self.shape
    out.prev_op = OPS.Flatten

    def _backward():
      self.grad += out.grad.reshape(original_shape)
    out._backward = _backward
    return out

  def unsqueeze(self, axis):
    out = Tensor(np.expand_dims(self.data, axis), _prev=(self,))
    out.prev_op = "UNSQUEEZE"

    def _backward():
      self.grad += np.squeeze(out.grad, axis=axis)
    
    out._backward = _backward
    return out

  def squeeze(self, axis=0):
    out = Tensor(np.squeeze(self.data, axis=axis), _prev=(self,), name="squeeze_out")
    original_shape = self.shape
    out.prev_op = OPS.Unsqueeze

    def _backward():
      self.grad += out.grad.reshape(original_shape)
    out._backward = _backward
    return out

  # Binary Ops
  def __add__(self, other):     return self.create_op(OPS.ADD, operands=(other,))
  def __mul__(self, other):     return self.create_op(OPS.MUL, operands=(other,))
  def __matmul__(self, other):  return self.dot(other)
  def dot(self, other):         return self.create_op(OPS.DOT, operands=(other,))
  def __pow__(self, other):     return self.create_op(OPS.POW, operands=(other,))
  def __radd__(self, other):    return self + other
  def __sub__(self, other):     return self + (-other)
  def __rsub__(self, other):    return other + (-self)
  def __rmul__(self, other):    return self * other
  def __truediv__(self, other): return self * other**-1
  def __rtruediv__(self, other):return other * self**-1

  def linear(self, weight: "Tensor", bias: Optional["Tensor"] = None):
    x = self * weight if len(weight.shape) == 1 else self.dot(weight)
    return x + bias if bias is not None else x

  def conv2d(self, weight: "Tensor", bias: "Tensor", in_channels: int, out_channels: int, stride: int = 1, padding: int = 0, debug=False):
    return self.create_op(
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
  # TODO: support add, mul with scalars
  def __neg__(self): return self * Tensor([-1], name="-1", requires_grad=False, device=self.device)
  def sqrt(self):    return self ** 0.5
  def relu(self):    return self.create_op(OPS.ReLU, )
  def softmax(self): return self.create_op(OPS.Softmax)
  def tanh(self):    return self.create_op(OPS.Tanh)
  def sigmoid(self): return self.create_op(OPS.Sigmoid)

  # Reduce Ops
  def mean(self, axis=None, keepdims=False):    return self.create_op(OPS.MEAN, forward_args=(axis, keepdims))
  def sum(self, axis=None, keepdims=False):     return self.create_op(OPS.SUM, forward_args=(axis, keepdims))
  def max(self, axis=None, keepdims=False):     return self.create_op(OPS.MAX, forward_args=(axis, keepdims))
  def min(self, axis=None, keepdims=False):     return self.create_op(OPS.MIN, forward_args=(axis, keepdims))
  def std(self, axis=None, keepdims=False):     return self.create_op(OPS.STD, forward_args=(axis, keepdims))
  def argmax(self, axis=None, keepdims=False):  return self.create_op(OPS.ARGMAX, forward_args=(axis, keepdims))
  def argmin(self, axis=None, keepdims=False):  return self.create_op(OPS.ARGMIN, forward_args=(axis, keepdims))
  def maxpool2d(self, filter=(2,2), stride=1):  return self.create_op(OPS.MaxPool2D, forward_args=(filter, stride))
  def avgpool2d(self, filter=(2,2), stride=1):  return self.create_op(OPS.AvgPool2D, forward_args=(filter, stride))
