#!/usr/bin/env python3
import numpy as np
from tensor import Tensor

# TODO: all losses need to be classes that inherit from this abstract one
class Loss:
  def __init__(self):
    pass

  def backward():
    pass
  
"""
class MSELoss(Loss):
  def __call__(self):
    # code that calculcates the loss value
"""

# z: network output, y: ground truth
# TODO: Tensor obj has no attribute __prev???
def MSELoss(z: Tensor, y: Tensor):
  assert (n := z.shape()[0]) == y.shape()[0]
  loss_val = 1/n * np.sum(np.power(z.data-y.data, 2))
  t = Tensor(loss_val, __children=z.__prev.copy())
  t.__prev.add(z)
  return t


if __name__ == '__main__':
  t1 = Tensor(np.random.rand(3))
  t2 = Tensor(np.random.rand(3))
  print(MSELoss(t1, t2))
