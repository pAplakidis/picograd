#!/usr/bin/env python3
import numpy as np

from picograd.tensor import Tensor
from picograd.loss import *
from picograd.optim import *
import picograd.nn as nn

class TestnetCLF(nn.Module):
  def __init__(self, in_feats, out_feats):
    super(TestnetCLF, self).__init__()
    self.dense1 = nn.Linear(in_feats, out_feats)

  def forward(self, x):
    x = self.dense1(x)
    return x.softmax()

def test_clf():
  t_in = Tensor(np.random.rand(10), name="t_in")
  print("t_in:", t_in)
  gt = Tensor(np.zeros((1,5)), name="ground_truth")  # for classification
  gt.data[0][1] = 1.0
  print(gt.data)

  model = TestnetCLF(t_in.shape([0]), 5)
  optim = SGD(model.get_params(), lr=1e-4)

  epochs = 200
  for i in range(epochs):
    print(f"[+] Epoch {i+1}/{epochs}")
    out = model(t_in)
    print("NN:", out.data)
    print("GT:", gt.data)

    loss = CrossEntropyLoss(out, gt)
    print("Loss:", loss.data[0])
    print()

    optim.reset_grad()
    loss.backward()
    optim.step()

def test_bclf():
  t_in = Tensor(np.random.rand(10), name="t_in")
  print("t_in:", t_in)
  gt = Tensor(np.ones((1,1)), name="ground_truth")  # for binary classification
  # loss = BCELoss(out, gt)

def test_reg():
  t_in = Tensor(np.random.rand(10), name="t_in")
  print("t_in:", t_in)
  gt = Tensor(np.random.rand(1, 5), name="ground_truth")  # for regression
  # loss = MSELoss(out, gt)


if __name__ == '__main__':
  test_clf()
