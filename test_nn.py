#!/usr/bin/env python3
import numpy as np

from tensor import Tensor
from loss import *
from optim import *
import nn

class Testnet(nn.Module):
  def __init__(self, in_feats, out_feats):
    super(Testnet, self).__init__()
    self.dense1 = nn.Linear(in_feats, out_feats)

  def forward(self, x):
    x = self.dense1(x)
    return x


if __name__ == '__main__':
  t_in = Tensor(np.random.rand(10), name="t_in")
  print("t_in:", t_in)
  #gt = Tensor(np.random.rand(1, 5), name="ground_truth")  # for regression
  #gt = Tensor(np.ones((1,5)) / 2, name="ground_truth")  # for classification
  gt = Tensor(np.ones((1,1)), name="ground_truth")  # for binary classification

  model = Testnet(t_in.shape([0]), 5)
  optim = SGD(model.get_params(), lr=1e-3)

  out = model(t_in)
  print(out)
  params = model.get_params()
  print(params)