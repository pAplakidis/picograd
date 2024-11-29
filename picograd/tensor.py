#!/usr/bin/env python3
import ctypes
import numpy as np
from sys import platform
from typing import Optional

from picograd.backend.ops import *
from picograd.util import *
from picograd.draw_utils import draw_dot

if platform == "linux" or platform == "linux2": PICOGRAD_LIB = ctypes.CDLL('./lib/libpicograd.so')  # linux
elif platform == "darwin":  PICOGRAD_LIB = ctypes.CDLL('./picograd/lib/libpicograd.dylib') # OS X
elif platform == "win32": PICOGRAD_LIB = ctypes.CDLL('./picograd/lib/libpicograd.dll') # Windows
else: PICOGRAD_LIB = None


class Tensor():
  def __init__(self, data: np.array, name: str = "t", _children: set = (), requires_grad: bool = True, verbose: bool = False):
    super().__init__()

    self._ctx = None  # TODO: use context like pytorch

    self.name = name
    self.data = data
    self.verbose = verbose

    self.requires_grad = requires_grad
    self.grad = np.zeros(self.data.shape) if self.requires_grad else None

    self._prev = set(_children)
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

  # TODO: cleanup like this (function + ops)
  # def __add__(self, other): return self._create_op_tensor(self.add(self.data, other.data))
  # def __sub__(self, other): return self._create_op_tensor(self.add(self.data, -other.data))
  # def __mul__(self, other): return self._create_op_tensor(self.mul(self.data, other.data))
  # def __pow__(self, other): return self._create_op_tensor(self.pow(self.data, other.data))
  # def __div__(self, other): return self._create_op_tensor(self * (other ** -1))

  def __add__(self, other):
    out = Tensor(self.data + other.data, _children=(self, other))
    out.prev_op = OPS.ADD

    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    return out

  def __mul__(self, other):
    out = Tensor(self.data * other.data, _children=(self, other))
    out.prev_op = OPS.MUL

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Tensor(self.data**other, _children=(self,))
    out.prev_op = OPS.POW

    def _backward():
      self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward
    return out

  def relu(self):
    out = Tensor(np.maximum(self.data, np.zeros(self.data.shape)),  _children=(self,))
    out.prev_op = OPS.ReLU

    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out

  def dot(self, other):
    out = Tensor(np.dot(self.data, other.data), _children=(self, other))
    out.prev_op = OPS.DOT

    def _backward():
      self.grad += np.dot(out.grad, other.data.T)
      other.grad += np.dot(self.data.T, out.grad)
    out._backward = _backward
    return out

  def __neg__(self): return self * Tensor(np.array(-1))
  def __radd__(self, other): return self + other
  def __sub__(self, other): return self + (-other)
  def __rsub__(self, other): return other + (-self)
  def __rmul__(self, other): return self * other
  def __truediv__(self, other): return self * other**-1
  def __rtruediv__(self, other): return other * self**-1

  def backward_recursive(self):
    # topological order all of the chidren in the graph
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

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
          if child not in visited:
            stack.append(child)
      elif v not in topo:
        topo.append(v)

    for node in reversed(topo):
      node._backward()
  
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

  def mean(self): return np.mean(self.data)

  def float(self):
    self.data = self.data.astype(np.float32)
    return self

  def long(self):
    self.data = self.data.astype(np.int64)
    return self

  def flatten(self):
    out = Tensor(self.data.flatten(), _children=(self,), name="flattenout")
    original_shape = self.data.shape
    out.prev_op = OPS.Flatten

    def _backward():
      self.grad += out.grad.reshape(original_shape)
    out._backward = _backward
    return out

  def unsqueeze(self, axis):
    out = Tensor(np.expand_dims(self.data, axis), _children=(self,))
    out.prev_op = "UNSQUEEZE"

    def _backward():
        self.grad += np.squeeze(out.grad, axis=axis)
    
    out._backward = _backward
    return out

  def squeeze(self, axis=0):
    out = Tensor(np.squeeze(self.data, axis=axis), _children=(self,), name="squeeze_out")
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

    for t0 in tmp:
      print("[==]", t0)
      if verbose:
        print("[data]\n", t0.data)
        print("[grad]\n", t0.grad)
        if t0.w:
          print("[w_data]\n", t0.w.data)
          print("[w_grad]\n", t0.w.grad)
        if t0.b:
          print("[b_data]\n", t0.b.data)
          print("[b_grad]\n", t0.b.grad)
      if t0.prev_op != None:
        print("====++++****++++====\n[OP]:", t0.prev_op ,"\n====++++****++++====")

  def linear(self, weight: "Tensor", bias: Optional["Tensor"] = None):
    x = self * weight if len(weight.shape) == 1 else self.dot(weight)
    return x + bias if bias is not None else x

  # FIXME: padding
  def conv2d(self, weight: np.ndarray, bias: np.ndarray, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, lib=PICOGRAD_LIB, debug=False):
    assert len(self.data.shape) == 3, "Conv2D input tensor must be 2D-RGB"
    assert kernel_size % 2 != 0, "Conv2D kenrel_size must be odd"

    self.kernel = weight
    self.b = bias

    _, H, W = self.data.shape # NOTE: double-check, we assume (c, h, w)
    H_out = ((H - kernel_size + 2*padding) // stride) + 1
    W_out = ((W - kernel_size + 2*padding) // stride) + 1

    out = Tensor(np.zeros((out_channels, H_out, W_out)), "conv2d_out", _children=self._prev.copy())
    out.data = out.data.astype(np.uint8)
    out._prev.append(self)
    out.prev_op = OPS.Conv2D

    self.grad = Tensor(np.zeros_like(self.data))
    out.grad = Tensor(np.zeros_like(out))

    def conv2d_cpp():
      assert lib is not None, "[Conv2D-CPP-ERROR] no .so library provided"
      print("Initializing c++ function")
      lib.conv2d.argtypes = [
          ctypes.c_int,                    # out_channels
          ctypes.c_int,                    # in_channels
          ctypes.c_int,                    # kernel_size
          ctypes.c_int,                    # padding
          ctypes.c_int,                    # H_out
          ctypes.c_int,                    # W_out
          ctypes.c_int,                    # H
          ctypes.c_int,                    # W
          ctypes.POINTER(ctypes.c_float),  # out.data
          ctypes.c_int,                    # len(out.data)
          ctypes.POINTER(ctypes.c_float),  # kernel.data 
          ctypes.c_int,                    # len(kernel.data)
          ctypes.POINTER(ctypes.c_float),  # b.data 
          ctypes.c_int,                    # len(b.data)
          ctypes.POINTER(ctypes.c_float),  # self.data 
          ctypes.c_int,                    # len(self.data)
      ]
      lib.conv2d.restype = ctypes.c_int

      print("Calling c++ function")
      result = lib.conv2d(
        out_channels, in_channels, kernel_size, padding, H_out, W_out, H, W,
        out.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(out.data),
        self.kernel.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(self.kernel.data),
        self.b.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(self.b.data),
        self.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(self.data)
      )
      print(result)
      return result

    # conv2d_cpp()
    # print("[+] C++ finished")

    for out_c in range(out_channels):
      for in_c in range(in_channels):
        i_idx = 0 - padding
        for i in range(H_out):
          j_idx = 0 - padding
          for j in range(W_out):
            # TODO: use something more simplified like this:
            # regn = padded_input[i_c, h:h + kernel_size, w:w + kernel_size]
            for k in range(kernel_size):
              for l in range(kernel_size):
                # handle padding
                if i_idx + k < 0 or j_idx + l < 0 or i_idx + k >= H or j_idx + l >= W:
                  out.data[out_c][i][j] += self.b.data[out_c]
                out.data[out_c][i][j] += self.data[in_c][i_idx + k][j_idx + l] * self.kernel.data[out_c][k][l] + self.b.data[out_c]
                if debug:
                  print(f"OUT ({out_c},{i},{j}), IN ({in_c},{i_idx},{j_idx}) => ({in_c},{i_idx+k},{j_idx+l}), W ({out_c},{k},{l})", end="(==)")
                  print(f"VAL: {out.data[out_c][i][j]}")
            if debug:
              print()
            j_idx += stride
          if debug:
            print()
          i_idx += stride
        if debug:
          print(f"IN_C {in_c}")
      if debug:
        print(f"OUT_C {out_c}")

    def _backward():
      # out.grad = np.ones_like(out.data)
      self.grad = np.zeros_like(self.data)
      self.kernel.grad = np.zeros_like(self.kernel.data)
      self.b.grad = np.sum(out.grad)

      for i in range(0, H, stride):
        for j in range(0, W, stride):
          self.grad[i:i+kernel_size, j:j+kernel_size] += out.grad * self.kernel.data
          self.kernel.grad = out.grad * self.data[i:i+kernel_size, j:j+kernel_size]
      out._backward = _backward

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

    out = Tensor(out_img, "maxpool2d", _children=self._prev.copy())
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

    out = Tensor(out_img, "avgpool2d", _children=self._prev.copy())
    out._prev.append(self)
    out.prev_op = OPS.AvgPool2D

    def _backward():
      self.grad = out.grad
    out._backward = _backward

    return out

  def tanh(self):
    t = (np.exp(2*self.data) - 1) / (np.exp(2*self.data) + 1)
    out = Tensor(t, name="tanh_out", _children=self._prev.copy())
    out._prev.append(self)
    out.prev_op = OPS.Tanh

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out

  def sigmoid(self):
    t = np.exp(self.data) / (np.exp(self.data) + 1)
    out = Tensor(t, name="sigmoid_out", _children=self._prev.copy())
    out._prev.append(self)
    out.prev_op = OPS.Sigmoid

    def _backward():
      self.grad = t * (1-t) * out.grad
    out._backward = _backward

    return out

  def softmax(self):
    exp_val = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
    probs = exp_val / np.sum(exp_val, axis=1, keepdims=True)
    out = Tensor(probs, name="softmax_out", _children=(self,))
    out.prev_op = OPS.Softmax

    def _backward():
      #self.grad += probs*(1-probs) * out.grad
      for i in range(out.data.shape[0]):
        for j in range(self.data.shape[0]):
          if i == j:
            self.grad[i] = (out.data[i] * (1-self.data[i])) * out.grad
          else:
            self.grad[i] = (-out.data[i] * self.data[j]) * out.grad
    out._backward = _backward

    return out
