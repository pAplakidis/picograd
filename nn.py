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
<<<<<<< HEAD
    self.weight = Tensor(np.random.rand(in_feats, out_feats))
    self.bias = Tensor(np.zeros(out_feats))
=======
    self.weight = Tensor(0.01 * np.random.rand(in_feats, out_feats))
    self.bias = Tensor(np.zeros((1, out_feats)))
>>>>>>> c1668d49e6cd83ae7b0c80ebd8c0d4900332c787

  def __call__(self, x: Tensor):
    return x.linear(self.weight, self.bias)
