#!/usr/bin/env python3
import numpy as np
from enum import Enum
from tensor import Tensor

class LayerType(Enum):
  NOLAYER = 0
  LINEAR = 1
  CONV2D = 2


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
    
class Linear(Layer):
  def __init__(self, in_feats, out_feats):
    self.type = LayerType['LINEAR'].value
    self.weight = Tensor(0.01 * np.random.rand(in_feats, out_feats), name="weight")
    self.bias = Tensor(np.zeros((1, out_feats)), name="bias")

  def __call__(self, x: Tensor):
    return x.linear(self.weight, self.bias)

# TODO: move most of the stuff from tensor here
class Conv2d(Layer):
  def __init__(self, kernel_size, stride=1):
    self.type = LayerType['CONV2D'].value

  def __call__(self, x: Tensor):
    pass
