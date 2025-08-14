#!/usr/bin/env python3
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd.tensor import Tensor
from picograd.draw_utils import draw_dot

if __name__ == "__main__":
  # TODO: test/debug gradients + requires_grad=False as well
  a = Tensor(np.array([1, 2, 3]))
  b = Tensor(np.array([4, 5, 6]))
  c = a + b
  print(c)
  print(c.data)
  c.backward()

  d = a * b
  print(d)
  print(d.data)
  d.backward()

  a = Tensor(np.array([1, 2, 3]))
  b = Tensor(np.array([4, 5, 6]))
  c = Tensor(np.array([7, 8, 9]))
  e = a * b + c
  print(e)
  print(e.data)
  e.backward()
  draw_dot(e, path="graphs/test_simple_ops/mul_add")

  a = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
  b = Tensor(np.array([[7, 8], [9, 10], [11, 12]]))
  c = Tensor(np.array([[13, 14], [15, 16]]))
  print(a.shape, b.shape)
  f = a.dot(b) + c
  print(f)
  print(f.data)
  f.backward()
  draw_dot(f, path="graphs/test_simple_ops/dot_add")

  x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
  w = Tensor(np.array([[7, 8], [9, 10], [11, 12]]))
  b = Tensor(np.array([[13, 14], [15, 16]]))
  print(a.shape, b.shape)
  f = (a.dot(w) + b).sigmoid()
  print(f)
  print(f.data)
  f.backward()
  draw_dot(f, path="graphs/test_simple_ops/loss")

