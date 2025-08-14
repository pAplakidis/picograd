#!/usr/bin/env python3
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd import Tensor
from picograd.backend.device import Devices, Device


t1 = np.random.randn(100, 50).astype(np.float32)
t2 = np.random.randn(50, 100).astype(np.float32)
t3 = np.random.randn(100, 100).astype(np.float32)

def test_linear(device):
  a = Tensor(t1, name="a", device=device)
  b = Tensor(t2, name="b", device=device)
  c = Tensor(t3, name="c", device=device)
  d = a.dot(b)
  d.backward()

  print(a.grad)
  print(b.grad)
  print(c.grad)
  print(d.grad)


if __name__== "__main__":
  test_linear(Device(Devices.CUDA))
  test_linear(Device(Devices.CPU))