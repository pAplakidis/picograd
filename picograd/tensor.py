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

# TODO: device abstraction layer
from picograd.backend.cuda.utils import *

from .draw_utils import draw_dot

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
    self.device= list(_prev)[0].device if len(list(_prev)) > 0 else device
    self._ctx = None  # TODO: use context like pytorch

    self.name = name
    self._shape = shape if data is None else data.shape
    self.debug = DEBUG
    self.verbose = bool(VERBOSE)
    self.requires_grad = requires_grad

    self._data = np.array(data, dtype=np.float32) if data is not None else np.zeros(self._shape)
    self._grad= np.zeros(self._shape) if requires_grad else None

    self._prev = set(_prev)
    self.prev_op = None
    self._backward = lambda: None

    self.layer = None
    self.w, self.b = None, None

    self.device_data = device_data
    self.device_grad = None
    if device.name != Devices.CPU: self.to(device)

  # FIXME: using data and grad internally (ops.py) causes cura errors
  # TODO: check what copy_data_to_host affects (data is moved when realized, so _data.shape might not be reliable)
  @property
  def data(self) -> np.ndarray:
    assert self._data is not None or self.device_data is not None, "Tensor data is not initialized."

    if self.device.name != Devices.CPU and self.device_data is not None:
      dev_data_to_host(self, free=False)

    if not isinstance(self._data, np.ndarray):
      self._data = np.array(self._data)
    return self._data
  
  @data.setter
  def data(self, value):
    initial_shape = self.shape
    self._data = value
    if self.device.name != Devices.CPU and self.device_data is not None:
      assert initial_shape == value.shape, "Tensor data shape does not match the initialized shape." # TODO: support new shapes (?)
      copy_data_to_device(self.device.manager, self.device_data, self._data)

  @property
  def grad(self):
    assert self._grad is not None or self.device_grad is not None, "Tensor grad is not initialized."

    if self.device.name != Devices.CPU and self.device_grad is not None:
      dev_grad_to_host(self, free=False)
    return self._grad

  @grad.setter
  def grad(self, value):
    initial_shape = self.shape
    self._grad = value
    # FIXME: this gets triggered in MNIST_simple/CUDA
    # assert initial_shape == value.shape, "Tensor data shape does not match the initialized shape." # TODO: support new shapes (?)
    if self.device.name != Devices.CPU and self.device_grad is not None:
      copy_data_to_device(self.device.manager, self.device_grad , self._grad) # FIXME: CUDA_ERROR_ILLEGAL_ADDRESS on relu_back after a few loops

  @property
  def dtype(self):
    return self.data.dtype if self._data is not None else np.float32

  @dtype.setter
  def dtype(self, value):
    if self._data is not None:
      self._data = self._data.astype(value)
    # if self.device_data is not None:
    #   self.device_data = self.device_data.astype(value)
    # if self._grad is not None:
    #   self._grad = self._grad.astype(value)
    # if self.device_grad is not None:
      # self.device_grad = self.device_grad.astype(value)

  @property
  def T(self): return Tensor(self.data.T, _prev=(self,), device=self.device)

  @property
  def item(self): return self.data

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

  def get_device_memory(self):
    """Returns the device memory of a tensor."""

    if self.device_data is None:
      print("Tensor is not on the CUDA device.")
      return

    size = self.data.nbytes
    host_buffer = np.empty(self.shape, dtype=self.data.dtype) # Create a host buffer to copy the data back to
    self.device.manager.memcpy_dtoh(
        host_buffer.ctypes.data,
        self.device_data,
        size
    )
    return host_buffer

  def to(self, device: Device):
    """Tranfers tensor to the specified device."""

    if device.name == Devices.CPU:
      if self.device_data is not None: tensor_to_host(self)
      self.device = device
      return self

    self.device = device
    if device.name == Devices.CUDA:
      # TODO: if result of CUDA op, no need to reallocate memory (op should return d_C as well)
      self._data = self._data.astype(np.float32)
      self._grad = self._grad.astype(np.float32)
      tensor_to_device(self)

    return self

  def __repr__(self):
    if self.verbose:
      return f"{color_yellow("Tensor")} (name={self.name}, shape={str(self.shape)}, device={str(self.device.name)}, data=\n{str(self.data)}\n, grad=\n{self.grad}, prev_op={self.prev_op}, prev_tensors={len(self._prev)})"
    else:
      return f"{color_yellow("Tensor")} (name={self.name}, shape={str(self.shape)}, device={str(self.device.name)}, data=\n{str(self.data)}\n, prev_op={self.prev_op}, prev_tensors={len(self._prev)})"
    
  def __del__(self):
    # if self.device_data is not None:
    #   free_device_tensor(self.device.manager, self.device_data)
    #   self.device_data = None

    # if self.device_grad is not None:
    #   free_device_tensor(self.device.manager, self.device_grad)
    #   self.device_grad = None
    pass

  def __getitem__(self, indices):
    return self.data[indices]

  def __setitem__(self, indices, value):
    self.data[indices] = value

  def __equal__(self, other): return np.equal(self.data, other.data)

  def __add__(self, other):
    self.func = Add(self.device.name)
    if self.device.name == Devices.CPU:
      out = Tensor(self.func.forward(self, other), _prev=(self, other))
    else:
      out = Tensor(
        device_data=self.func.forward(self, other),
        shape=self.shape,
        _prev=(self, other),
        device=self.device
      )
    out.prev_op = OPS.ADD
    out._backward = lambda: self.func.backward(out.grad)
    return out

  def __mul__(self, other):
    self.func = Mul(self.device.name)
    if self.device.name == Devices.CPU:
      out = Tensor(self.func.forward(self, other), _prev=(self, other))
    else:
      out = Tensor(
        device_data=self.func.forward(self, other),
        shape=self.shape,
        _prev=(self, other),
        device=self.device  
      )
    out.prev_op = OPS.MUL
    out._backward = lambda: self.func.backward(out.grad)
    return out

  def __matmul__(self, other): return self.dot(other)

  def dot(self, other):
    self.func = Dot(self.device.name)
    if self.device.name == Devices.CPU:
      out = Tensor(self.func.forward(self, other), _prev=(self, other))
    else:
      out = Tensor(
        device_data=self.func.forward(self, other),
        shape=(self.shape[0], other.shape[1]),
        _prev=(self, other),
        device=self.device
      )
    out.prev_op = OPS.DOT
    out._backward = lambda: self.func.backward(out.grad)
    return out

  # TODO: implement these in ops (cpu and cuda)
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Tensor(self.data**other, _prev=(self,), device=self.device)
    out.prev_op = OPS.POW

    def _backward():
      self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward
    return out

  def relu(self):
    self.func = ReLU(self.device.name)
    if self.device.name == Devices.CPU:
      out = Tensor(self.func.forward(self), _prev=(self,), device=self.device)
    else:
      out = Tensor(
        device_data=self.func.forward(self),
        shape=self.shape,
        _prev=(self,),
        device=self.device
      )
    out.prev_op = OPS.ReLU
    out._backward = lambda: self.func.backward(out.grad)
    return out

  def __neg__(self): return self * Tensor(np.array(-1), device=self.device)
  def __radd__(self, other): return self + other
  def __sub__(self, other): return self + (-other)
  def __rsub__(self, other): return other + (-self)
  def __rmul__(self, other): return self * other
  def __truediv__(self, other): return self * other**-1
  def __rtruediv__(self, other): return other * self**-1

  def backward(self):
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
    for node in reversed(topo): node._backward()
  
  # TODO: move to ops
  def mean(self): return Tensor(np.mean(self.data), _prev=(self,))

  def float(self):
    self.data = self.data.astype(np.float32)
    return self

  def long(self):
    self.data = self.data.astype(np.int64)
    return self

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
  
  def linear(self, weight: "Tensor", bias: Optional["Tensor"] = None):
    x = self * weight if len(weight.shape) == 1 else self.dot(weight)
    return x + bias if bias is not None else x

  def conv2d(self, weight: "Tensor", bias: "Tensor", in_channels: int, out_channels: int, stride: int = 1, padding: int = 0, debug=False):
    self.func = Conv2D(self.device.name)
    if self.device.name == Devices.CPU:
      out = Tensor(
        self.func.forward(self, weight, bias,  in_channels, out_channels, stride, padding),
        _prev=(self, weight, bias),
      )
    else:
      out = Tensor(
        device_data=self.func.forward(self, weight, bias,  in_channels, out_channels, stride, padding),
        shape= (
          self.shape[0], out_channels, 
          (self.shape[2] - weight.shape[2] + 2 * padding) // stride + 1,
          (self.shape[3] - weight.shape[3] + 2 * padding) // stride + 1
        ),
        _prev=(self, weight, bias),
        device=self.device
      )
    out.prev_op = OPS.Conv2D
    out._backward = lambda: self.func.backward(out.grad)
    return out

  def batchnorm1d(self):
    pass

  def batchnorm2d(self):
    pass

  def maxpool2d(self, filter=(2,2), stride=1):
    # TODO: assert dimensionality
    # TODO: double-check and test
    channels, height, width = self.shape
    out_height = (height - filter[0]) // stride + 1
    out_width = (width - filter[1]) // stride + 1
    out_img = np.zeros((channels, out_height, out_width))
    mask = np.zeros_like(self.data)

    for i in range(out_height):
      for j in range(out_width):
        h_start, h_end = i * stride, i * stride+ filter[0]
        w_start, w_end = j * stride, j * stride + filter[1]

        # Extract the region to perform max pooling
        x_slice = self.data[:, h_start:h_end, w_start:w_end]

        # Perform max pooling along the specified axis
        max_values = np.max(x_slice, axis=(1, 2), keepdims=True)

        # Update the output and mask
        out_img[:, i, j] = max_values[:, 0, 0]
        mask[:, h_start:h_end, w_start:w_end] = (x_slice == max_values)

    out = Tensor(out_img, "maxpool2d", _prev=self._prev.copy())
    out._prev.append(self)
    out.prev_op = OPS.MaxPool2D

    # TODO: implement backward
    def _backward():
      self.grad = out.grad
    out._backward = _backward

    return out

  # TODO: fix this like maxpool2D
  def avgpool2d(self, filter=(2,2), stride=1, padding=0):
    # TODO: assert dimensionality
    # TODO: handle channels and padding as well
    # TODO: double-check if stride is used correctly

    # pooling out_dim = (((input_dim - filter_dim) / stride) + 1) * channels
    out_img = np.ones(((self.shape[0] - filter[0] // stride) + 1, (self.shape[1] - filter[1] // stride) + 1))
    for i in range(0, self.shape[0]-filter[0], filter[0]):
      for j in range(0, self.shape[1]-filter[0], filter[1]):
        tmp = []
        for n in range(filter[0]):
          for m in range(filter[1]):
            # TODO: keep pooling (max) indices, for use on GANs (like SegNet)
            tmp.append(out_img[i*stride+n][j*stride+m])
        out_img[i][j] = np.array(tmp).mean()

    out = Tensor(out_img, "avgpool2d", _prev=self._prev.copy())
    out._prev.append(self)
    out.prev_op = OPS.AvgPool2D

    def _backward():
      self.grad = out.grad
    out._backward = _backward

    return out

  def tanh(self):
    t = (np.exp(2*self.data) - 1) / (np.exp(2*self.data) + 1)
    out = Tensor(t, name="tanh_out", _prev=self._prev.copy())
    out._prev.append(self)
    out.prev_op = OPS.Tanh

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out

  def sigmoid(self):
    t = np.exp(self.data) / (np.exp(self.data) + 1)
    out = Tensor(t, name="sigmoid_out", _prev=(self,))
    out.prev_op = OPS.Sigmoid

    def _backward():
      self.grad = t * (1-t) * out.grad
    out._backward = _backward

    return out

  def softmax(self):
    self.func = Softmax(self.device.name)
    if self.device.name == Devices.CPU:
      out = Tensor(self.func.forward(self), name="softmax_out", _prev=(self,))
    else:
      out = Tensor(
        device_data=self.func.forward(self),
        shape=self.shape,
        _prev=(self,),
        device=self.device
      )
    out.prev_op = OPS.Softmax
    out._backward = lambda: self.func.backward(out.grad)
    return out
