#!/usr/bin/env python3
import numpy as np
from tensor import Tensor

class Module:
  def __init__(self):
    pass

  def forward(self):
    pass

class Linear:
  def __init__(self, in_feats, out_feats):
    self.weight = Tensor(np.random.rand(in_feats, out_feats))
    self.bias = Tensor(np.zeros(out_feats))

  def __call__(self, x: Tensor):
    return x.linear(self.weight, self.bias)
