#!/usr/bin/env python3
import numpy as np
from tensor import Tensor

def MSELoss(z: Tensor, y: Tensor):
  assert (n := z.shape()[0]) == y.shape()[0]
  return 1/n * np.sum(np.power(z.data-y.data, 2))


if __name__ == '__main__':
  t1 = Tensor(np.random.rand(3))
  t2 = Tensor(np.random.rand(3))
  print(MSELoss(t1, t2))
