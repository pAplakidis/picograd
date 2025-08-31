#!/usr/bin/env python3
from time import time
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd.tensor import Tensor
from picograd.util import is_cuda_available
from picograd.backend.device import Devices, Device

# device = Device(Devices.CPU)
device = Device(Devices.CUDA) if is_cuda_available() else Device(Devices.CPU)
print("[*] Using device", device.name, "\n")

def generate_tensors():
  t1 = np.random.randn(50, 100)
  return Tensor(t1, name="a", device=device)

def test_reduce_ops():
  a = generate_tensors()
  print(a.max())
  print(a.min())
  print(a.sum())
  print(a.mean())
  print(a.std())
  print(a.argmax())
  print(a.argmin())

  print(a.max(axis=1))
  print(a.min(axis=1))
  print(a.sum(axis=1))
  print(a.mean(axis=1))
  print(a.std(axis=1))
  print(a.argmax(axis=1))
  print(a.argmin(axis=1))


if __name__ == "__main__":
  test_reduce_ops()
