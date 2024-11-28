#!/usr/bin/env python3
import numpy as np
from enum import Enum, auto
from picograd.tensor import Tensor
from picograd.backend.ops import *

class LayerType(Enum):
  NOLAYER = auto()
  LINEAR = auto()
  CONV2D = auto()
  MAXPOOL2D = auto()
  AVGPOOL2D = auto()

  def __str__(self):
    return self.name

# TODO: Sequential

class Module:
  def __init__(self):
    self.params = []
    self.train = True

  def train(self):
    self.train = True
    return self

  def eval(self):
    self.eval = False
    return self

  def forward(self):
    return None

  def __call__(self, *params):
    return self.forward(*params)

  def get_params(self):
    # TODO: params are tensors(weights, biases, etc) not layers
    for name, param in self.__dict__.items():
      if isinstance(param, Layer):
        self.params.append(param)
    return self.params


class Layer:
  def __init__(self):
    self.type = None
    self.t_in = None
    self.t_out = None
    self.weight = None
    self.bias = None


class Linear(Layer):
  def __init__(self, in_feats: int, out_feats: int):
    self.type = LayerType.LINEAR
    self.weight = Tensor(0.01 * np.random.rand(in_feats, out_feats), name="weight")
    self.bias = Tensor(np.zeros((1, out_feats)), name="bias")

  def __call__(self, x: Tensor):
    self.t_in = x
    self.t_out = x.linear(self.weight, self.bias)
    return self.t_out


class Conv2d(Layer):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0):
    self.type = LayerType.CONV2D

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    self.weight = Tensor(np.random.randint(0, 255, (out_channels, kernel_size, kernel_size), dtype=np.uint8), "conv2D_kernel")  # weight
    self.bias = Tensor(np.zeros((out_channels, 1, 1)), name="bias")

  def __call__(self, x: Tensor):
    self.t_in = x
    self.t_out = x.conv2d(self.weight, self.bias, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, debug=False)
    return self.t_out


class MaxPool2D(Layer):
  def __init__(self, filter=(2,2), stride=1):
    self.type = LayerType.MAXPOOL2D
    
    self.filter = filter
    self.stride = stride

  def __call__(self, x: Tensor):
    self.t_in = x
    self.t_out = x.maxpool2d(self.filter, self.stride)
    return self.t_out


class AvgPool2D(Layer):
  def __init__(self, filter=(2,2), stride=1, padding=0):
    self.type = LayerType.AVGPOOL2D
    
    self.filter = filter
    self.stride = stride
    self.padding = padding

  def __call__(self, x: Tensor):
    self.t_in = x
    self.t_out = x.avgpool2d(self.filter, self.stride, self.padding)
    return self.t_out
