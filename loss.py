#!/usr/bin/env python3
import numpy as np
from tensor import *

# TODO: all losses need to be classes that inherit from this abstract one
"""
class Loss:
  def __init__(self):
    pass

class MSELoss(Loss):
  def __call__(self):
    # code that calculcates the loss value
"""

# z: network output, y: ground truth
# Mean Squared Error Loss
def MSELoss(z: Tensor, y: Tensor):
  assert (n := z.shape()[0]) == y.shape()[0], f"Z Tensor doesn't have the same shape as ground-truth Y: z.shape={str(z.data.shape)}, y.shape={str(y.data.shape)}"
  loss_val = 1/n * np.sum((z.data-y.data) ** 2)
  t = Tensor(loss_val, name="mseloss_out", _children=z._prev.copy())
  t._prev.append(z)
  t.prev_op = OPS["MSELoss"]
  return t

# Mean Absolute Error Loss
def MAELoss(z: Tensor, y: Tensor):
  assert (n := z.shape()[0]) == y.shape()[0], f"Z Tensor doesn't have the same shape as ground-truth Y: z.shape={str(z.data.shape)}, y.shape={str(y.data.shape)}"
  loss_val = 1/n * np.sum(np.abs(z.data-y.data))
  t = Tensor(loss_val, name="maeloss_out", _children=z._prev.copy())
  t._prev.append(z)
  t.prev_op = OPS["MAELoss"]
  return t

# Binary Cross Entropy Loss
def BCELoss():
  pass

# Categorical Cross Entropy
def CrossEntropyLoss():
  pass

# Negative Log Likelihood Loss
def NLLLoss():
  pass


if __name__ == '__main__':
  t1 = Tensor(np.random.rand(3))
  t2 = Tensor(np.random.rand(3))
  print(MSELoss(t1, t2))
