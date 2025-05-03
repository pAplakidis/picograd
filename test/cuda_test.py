#!/usr/bin/env python3
import unittest
from time import time
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd.tensor import Tensor
from picograd.backend.device import Devices, Device

from picograd.backend.cuda.utils import *

DEBUG = int(os.getenv("DEBUG", 0))


# TODO: use unittest
# FIXME: device ops cannot be consecutive - segfaults undeterministically (probably due to memory management)
if __name__ == "__main__":
  device = Device(Devices.CUDA, debug=DEBUG)
  print("[*] Using device", device.name)

  a = Tensor(np.random.randn(100,100), requires_grad=False, device=device)
  print(a)

  b = Tensor(np.random.randn(100, 100), requires_grad=False).to(device)
  print(b)

  # c = a + b
  # print(c)
  # assert np.allclose(c.data, a.data + b.data), "CUDA addition failed"
  # print("[+] Add OK")

  # c = a * b
  # print(c)
  # assert np.allclose(c.data, a.data * b.data), "CUDA multiplication failed"
  # print("[+] Mul OK")

  c = a.dot(b)
  print(c)
  assert np.allclose(c.data, a.data @ b.data, atol=1e-4), "CUDA dot product failed"
  print("[+] Dot OK")

  print("[+] Cuda test OK")
