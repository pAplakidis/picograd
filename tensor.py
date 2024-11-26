#!/usr/bin/env python3
import ctypes
import numpy as np
from sys import platform

from backend.ops import *
from util import *
from draw_utils import draw_dot

if platform == "linux" or platform == "linux2": PICOGRAD_LIB = ctypes.CDLL('./lib/libpicograd.so')  # linux
elif platform == "darwin":  PICOGRAD_LIB = ctypes.CDLL('./lib/libpicograd.dylib') # OS X
elif platform == "win32": PICOGRAD_LIB = ctypes.CDLL('./lib/libpicograd.dll') # Windows
else: PICOGRAD_LIB = None

# TODO: add a .add_to_prev() to prevent duplicate code
class Tensor:
  def __init__(self, data: np.array, name="t", _children=[], verbose=False):
    self.name = name
    self.data = data
    self.verbose = verbose

    self._ctx = None  # TODO: use context like pytorch
    self._prev = list(_children)
    self.grad = np.ones(self.data.shape)
    self.out = None
    self.prev_op = None
    self._backward = lambda: None

    self.layer = None
    self.w, self.b = None, None

  def __repr__(self):
    if self.verbose:
      return f"Tensor(name={self.name}, shape={str(self.shape())}, data={str(self.data)}, grad={self.grad}), prev_op={get_key_from_value(OPS, self.prev_op)}, prev_tensors={len(self._prev)})"
    else:
      return f"Tensor(name={self.name}, shape={str(self.shape())}, prev_op={get_key_from_value(OPS, self.prev_op)}, prev_tensors={len(self._prev)})"

  # TODO: support item assignment as well
  def __getitem__(self, idx):
    if isinstance(idx, tuple):
      # Recursively access the elements
      result = self.data
      for i in idx:
          result = result[i]
      return result
    else:
        return self.data[idx]

  def _create_op_tensor(self, data):
    children = self._prev.copy()
    children.append(self)

    # TODO: add small backward here (for later, when we have a more detailed graph)
    return Tensor(data, _children=children)

  # TODO: remove commeneted when proper recursive backward() is implemented
  # TODO: implement backward for each op (?)
  def __add__(self, other): return self._create_op_tensor(self.data + other.data)
  def __sub__(self, other): return self._create_op_tensor(self.data - other.data)
  def __mul__(self, other): return self._create_op_tensor(self.data * other.data)
  def __pow__(self, other): return self._create_op_tensor(self.data ** other.data)
  def __div__(self, other): return self._create_op_tensor(self * (other ** -1))
  def dot(self, other): return self._create_op_tensor(np.dot(self.data, other.data))
  def T(self): return Tensor(self.data.T)

  # TODO: use @property
  def item(self): return self.data

  # TODO: use @property
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
    self.out = Tensor(self.data.flatten(), name="flattenout")
    self.out.prev_channels, self.out.prev_height, self.out.prev_width = self.data.shape
    self.out._prev = self._prev.copy()
    self.out._prev.append(self)
    self.out.prev_op = OPS.Flatten

    def _backward():
      self.grad = self.out.grad
      self.data = self.out.data.copy().reshape(self.out.prev_channels, self.out.prev_height, self.out.prev_width)
    self._backward = _backward

    return self.out

  # TODO: backward + squeeze
  def unsqueeze(self, axis=0):
    self.out = Tensor(np.expand_dims(self.data, axis=axis), name="unsqueeze_out")
    # self.out.prev_channels, self.out.prev_height, self.out.prev_width = self.data.shape
    self.out._prev = self._prev.copy()
    self.out._prev.append(self)
    self.out.prev_op = OPS.Unsqueeze

    def _backward():
      self.grad = self.out.grad
      # self.data = self.out.data.copy().reshape(self.out.prev_channels, self.out.prev_height, self.out.prev_width)
    self._backward = _backward

    return self.out

  def squeeze(self):
    pass

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
        print("====++++****++++====\n[OP]:", get_key_from_value(OPS, t0.prev_op) ,"\n====++++****++++====")

  def linear(self, w, b):
    self.w = w
    self.b = b
    self.out = self.dot(self.w.data) + self.b.data
    self.out.name = "linearout"
    self.out._prev = self._prev.copy()
    self.out._prev.append(self)
    self.out.prev_op = OPS.Linear

    # TODO: won't be needed if we implement low level ops
    def _backward():
      #print(self.data.shape, self.out.grad.shape)
      if len(self.data.shape) == 1:
        self.w.grad = np.dot(self.data[np.newaxis].T, self.out.grad)
      else:
        self.w.grad = np.dot(self.data.T, self.out.grad)

      #print(self.out.grad.shape)
      self.b.grad = np.sum(self.out.grad, axis=0, keepdims=True)

      # print(self.out.grad.shape, self.w.data.shape)
      self.grad = np.dot(self.out.grad, self.w.data.T)

      # TODO: does this fix the inf/vanishing-gradient problem or just hide it?
      grad_norm = np.linalg.norm(self.grad)
      if grad_norm > MAX_GRAD_NORM:
        self.grad = self.grad * (MAX_GRAD_NORM / grad_norm)

    self.out._backward = _backward
    return self.out

  # FIXME: padding
  def conv2d(self, weight: np.ndarray, bias: np.ndarray, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, lib=PICOGRAD_LIB, debug=False):
    assert len(self.data.shape) == 3, "Conv2D input tensor must be 2D-RGB"
    assert kernel_size % 2 != 0, "Conv2D kenrel_size must be odd"

    self.kernel = weight
    self.b = bias

    _, H, W = self.data.shape # NOTE: double-check, we assume (c, h, w)
    H_out = ((H - kernel_size + 2*padding) // stride) + 1
    W_out = ((W - kernel_size + 2*padding) // stride) + 1

    self.out = Tensor(np.zeros((out_channels, H_out, W_out)), "conv2d_out", _children=self._prev.copy())
    self.out.data = self.out.data.astype(np.uint8)
    self.out._prev.append(self)
    self.out.prev_op = OPS.Conv2D

    self.grad = Tensor(np.zeros_like(self.data))
    self.out.grad = Tensor(np.zeros_like(self.out))

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
        self.out.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(self.out.data),
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
                  self.out.data[out_c][i][j] += self.b.data[out_c]
                self.out.data[out_c][i][j] += self.data[in_c][i_idx + k][j_idx + l] * self.kernel.data[out_c][k][l] + self.b.data[out_c]
                if debug:
                  print(f"OUT ({out_c},{i},{j}), IN ({in_c},{i_idx},{j_idx}) => ({in_c},{i_idx+k},{j_idx+l}), W ({out_c},{k},{l})", end="\self (==)")
                  print(f"VAL: {self.out.data[out_c][i][j]}")
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
      # self.out.grad = np.ones_like(self.out.data)
      self.grad = np.zeros_like(self.data)
      self.kernel.grad = np.zeros_like(self.kernel.data)
      self.b.grad = np.sum(self.out.grad)

      for i in range(0, H, stride):
        for j in range(0, W, stride):
          self.grad[i:i+kernel_size, j:j+kernel_size] += self.out.grad * self.kernel.data
          self.kernel.grad = self.out.grad * self.data[i:i+kernel_size, j:j+kernel_size]
      self.out._backward = _backward

    return self.out

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

    self.out = Tensor(out_img, "maxpool2d", _children=self._prev.copy())
    self.out._prev.append(self)
    self.out.prev_op = OPS.MaxPool2D

    # TODO: implement backward
    def _backward():
      self.grad = self.out.grad
    self.out._backward = _backward

    return self.out

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

    self.out = Tensor(out_img, "avgpool2d", _children=self._prev.copy())
    self.out._prev.append(self)
    self.out.prev_op = OPS.AvgPool2D

    def _backward():
      self.grad = self.out.grad
    self.out._backward = _backward

    return self.out

  # TODO: backward needs to implemented for all tensors in each op (a = b + c => a.back -> b.back and c.back)
  # that's why deepwalk is implemented
  def deep_walk(self):
    def walk(node, visited, nodes):
      if node._ctx:
        [walk(i, visited, nodes) for i in node._ctx.parents if i not in visited]
        nodes.append(node)
      return nodes
    return walk(self, set(), [])

  # TODO: maybe implement a backward for each type of op instead of layer??
  # TODO: do we need reversed?? (double check, since we start from loss and backward)
  def backward(self):
    #self.grad = np.ones(self.data.shape)
    if self.verbose:
      print("\n[+] Before backpropagation")
      self.print_graph()
    draw_dot(self)

    self._backward()
    for t0 in reversed(list(self._prev)):
      t0._backward()

    if self.verbose:
      print("\n[+] After backpropagation")
      self.print_graph()
    draw_dot(self)

  def relu(self):
    self.out = Tensor(np.maximum(self.data, np.zeros(self.data.shape)), name="relu_out", _children=self._prev.copy())
    self.out._prev.append(self)
    self.out.prev_op = OPS.ReLU

    def _backward():
      self.grad += self.out.grad * (self.data > 0)
    self.out._backward = _backward

    return self.out

  def tanh(self):
    t = (np.exp(2*self.data) - 1) / (np.exp(2*self.data) + 1)
    self.out = Tensor(t, name="tanh_out", _children=self._prev.copy())
    self.out._prev.append(self)
    self.out.prev_op = OPS.Tanh

    def _backward():
      self.grad += (1 - t**2) * self.out.grad
    self.out._backward = _backward

    return self.out

  def sigmoid(self):
    t = np.exp(self.data) / (np.exp(self.data) + 1)
    self.out = Tensor(t, name="sigmoid_out", _children=self._prev.copy())
    self.out._prev.append(self)
    self.out.prev_op = OPS.Sigmoid

    def _backward():
      self.grad = t * (1-t) * self.out.grad
    self.out._backward = _backward

    return self.out

  def softmax(self):
    exp_val = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
    probs = exp_val / np.sum(exp_val, axis=1, keepdims=True)
    self.out = Tensor(probs, name="softmax_out", _children=self._prev.copy())
    self.out._prev.append(self)
    self.out.prev_op = OPS.Softmax

    def _backward():
      #self.grad += probs*(1-probs) * self.out.grad
      for i in range(self.out.data.shape[0]):
        for j in range(self.data.shape[0]):
          if i == j:
            self.grad[i] = (self.out.data[i] * (1-self.data[i])) * self.out.grad
          else:
            self.grad[i] = (-self.out.data[i] * self.data[j]) * self.out.grad
    self.out._backward = _backward

    return self.out
