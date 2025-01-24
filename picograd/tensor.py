#!/usr/bin/env python3
import ctypes
import numpy as np
from sys import platform
from typing import Optional

from picograd.backend.function import *
from picograd.backend.ops import *
from picograd.util import *

from .draw_utils import draw_dot

# init c++ library
if platform == "linux" or platform == "linux2": PICOGRAD_LIB = ctypes.CDLL('./lib/libpicograd.so')  # linux
elif platform == "darwin":  PICOGRAD_LIB = ctypes.CDLL('./picograd/lib/libpicograd.dylib') # OS X
elif platform == "win32": PICOGRAD_LIB = ctypes.CDLL('./picograd/lib/libpicograd.dll') # Windows
else: PICOGRAD_LIB = None


class Tensor():
  def __init__(self, data: np.array, name: str = "t", _prev: set = (), requires_grad: bool = True, verbose: bool = False):
    super().__init__()

    self._ctx = None  # TODO: use context like pytorch

    self.name = name
    self.data = data
    self.verbose = verbose

    self.requires_grad = requires_grad
    self.grad = np.zeros(self.data.shape) if self.requires_grad else None

    self._prev = set(_prev)
    self.prev_op = None
    self._backward = lambda: None

    self.layer = None
    self.w, self.b = None, None

  def __repr__(self):
    if self.verbose:
      return f"Tensor(name={self.name}, shape={str(self.shape)}, data={str(self.data)}, grad={self.grad}, prev_op={self.prev_op}, prev_tensors={len(self._prev)})"
    else:
      return f"Tensor(name={self.name}, shape={str(self.shape)}, prev_op={self.prev_op}, prev_tensors={len(self._prev)})"

  def __getitem__(self, indices):
    return self.data[indices]

  def __setitem__(self, indices, value):
    self.data[indices] = value

  def __equal__(self, other): return np.equal(self.data, other.data)

  # FIXME: graph shows only last bit since we used function.py
  def __add__(self, other):
    self.func = Add()
    out = Tensor(self.func.forward(self, other), _prev=(self, other))
    out.prev_op = OPS.ADD
    out._backward = lambda: self.func.backward(out.grad)
    return out

  def __mul__(self, other):
    self.func = Mul()
    out = Tensor(self.func.forward(self, other), _prev=(self, other))
    out.prev_op = OPS.MUL
    out._backward = lambda: self.func.backward(out.grad)
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Tensor(self.data**other, _prev=(self,))
    out.prev_op = OPS.POW

    def _backward():
      self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward
    return out

  def relu(self):
    out = Tensor(np.maximum(self.data, np.zeros(self.data.shape)),  _prev=(self,))
    out.prev_op = OPS.ReLU

    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out

  def dot(self, other):
    self.func = Dot()
    out = Tensor(self.func.forward(self, other), _prev=(self, other))
    out.prev_op = OPS.DOT
    out._backward = lambda: self.func.backward(out.grad)
    return out

  def __neg__(self): return self * Tensor(np.array(-1))
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
  
  @property
  def T(self): return Tensor(self.data.T)

  @property
  def item(self): return self.data

  @property
  def shape(self, idxs=None):
    if idxs is None:
      return self.data.shape
    ret = []
    shp = self.data.shape
    for idx in idxs:
      ret.append(shp[idx])
    
    if len(ret) == 1:
      ret = int(ret[0])
    return ret

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
    original_shape = self.data.shape
    out.prev_op = OPS.Reshape

    def _backward():
      self.grad += out.grad.reshape(original_shape)
    out._backward = _backward
    return out

  def flatten(self):
    out = Tensor(self.data.flatten(), _prev=(self,), name="flattenout")
    original_shape = self.data.shape
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
    original_shape = self.data.shape
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
      if verbose:
        print("[data]\n", node.data)
        print("[grad]\n", node.grad)
      if node.prev_op != None:
        print("====++++****++++====\n[OP]:", node.prev_op ,"\n====++++****++++====")

  
  def linear(self, weight: "Tensor", bias: Optional["Tensor"] = None):
    x = self * weight if len(weight.shape) == 1 else self.dot(weight)
    return x + bias if bias is not None else x

  def conv2d(self, weight: "Tensor", bias: "Tensor", in_channels: int, out_channels: int, stride: int = 1, padding: int = 0, debug=False):
    self.func = Conv2D()
    out = Tensor(self.func.forward(self, weight, bias,  in_channels, out_channels, stride, padding), _prev=(self, weight, bias))
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
    channels, height, width = self.data.shape
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
    out_img = np.ones(((self.data.shape[0] - filter[0] // stride) + 1, (self.data.shape[1] - filter[1] // stride) + 1))
    for i in range(0, self.data.shape[0]-filter[0], filter[0]):
      for j in range(0, self.data.shape[1]-filter[0], filter[1]):
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
    exp_val = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
    probs = exp_val / np.sum(exp_val, axis=1, keepdims=True)
    out = Tensor(probs, name="softmax_out", _prev=(self,))
    out.prev_op = OPS.Softmax

    def _backward():
      self.grad = np.zeros_like(out.data)
      for i in range(out.shape[0]):
        s = out.data[i].reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        self.grad[i] = np.dot(jacobian, out.grad[i])
    out._backward = _backward
    return out
