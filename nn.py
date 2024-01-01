#!/usr/bin/env python3
import numpy as np
from enum import Enum
from tensor import Tensor

class LayerType(Enum):
  NOLAYER = 0
  LINEAR = 1
  CONV2D = 2
  MAXPOOL2D = 3
  AVGPOOL2D = 4


class Module:
  def __init__(self):
    self.params = []

  def forward(self):
    return

  def __call__(self, *params):
    self.forward(*params)

  def get_params(self):
    # TODO: add params from each layer (w,b)
    return self.params


class Layer:
  def __init__(self):
    self.type = None
    self.weight = None
    self.bias = None
    

class Linear(Layer):
  def __init__(self, in_feats: int, out_feats: int):
    self.type = LayerType['LINEAR'].value
    self.weight = Tensor(0.01 * np.random.rand(in_feats, out_feats), name="weight")
    self.bias = Tensor(np.zeros((1, out_feats)), name="bias")

  def __call__(self, x: Tensor):
    return x.linear(self.weight, self.bias)


class Conv2d(Layer):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0):
    self.type = LayerType['CONV2D'].value

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    self.weight = Tensor(np.random.randint(0, 255, (out_channels, kernel_size, kernel_size), dtype=np.uint8), "conv2D_kernel")  # weight
    self.bias = Tensor(np.zeros((out_channels, 1, 1)), name="bias")

  def __call__(self, x: Tensor):
    return x.conv2d(self.weight, self.bias, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, debug=False)


class MaxPool2D(Layer):
  def __init__(self, filter=(2,2), stride=1):
    self.type = LayerType['MAXPOOL2D'].value
    
    self.filter = filter
    self.stride = stride

  def __call__(self, x: Tensor):
    return x.maxpool2d(self.filter, self.stride)


class AvgPool2D(Layer):
  def __init__(self, filter=(2,2), stride=1, padding=0):
    self.type = LayerType['AVGPOOL2D'].value
    
    self.filter = filter
    self.stride = stride
    self.padding = padding

  def __call__(self, x: Tensor):
    return x.avgpool2d(self.filter, self.stride, self.padding)
