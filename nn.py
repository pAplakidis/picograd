#!/usr/bin/env python3
import numpy as np
from tensor import Tensor

class Module:
  def __init__(self):
    self.params = []

  def forward(self):
    return

  def __call__(self, *params):
    self.forward(*params)

class Linear:
  def __init__(self, in_feats, out_feats):
    self.weight = Tensor(0.01 * np.random.rand(in_feats, out_feats), name="weight")
    self.bias = Tensor(np.zeros((1, out_feats)), name="bias")

  def __call__(self, x: Tensor):
    return x.linear(self.weight, self.bias)
