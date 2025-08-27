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
  t1 = np.random.randn(100, 50).astype(np.float32)
  a = Tensor(t1, name="a", device=device)
  b = Tensor(t1, name="b", device=Device(Devices.CPU))
  return a, b, t1

def test_relu():
  a, b, t1 = generate_tensors()
  res = a.relu()
  res.backward()

  assert np.allclose(res.data, np.maximum(0, t1)), "ReLU failed"
  print("[+] ReLU OK")

def test_softmax():
  a, b, _ = generate_tensors()
  res_a = a.softmax()
  res_a.backward()

  assert np.allclose(res_a.data, b.softmax().data), "Softmax failed"
  print("[+] Softmax OK")

if __name__ == "__main__":
  test_relu()
  test_softmax()
