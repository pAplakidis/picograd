#!/usr/bin/env python3
import unittest
from time import time
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd.tensor import Tensor
from picograd.backend.device import Device


# TODO: use unittest
if __name__ == "__main__":
  a = Tensor(np.random.randn(100), requires_grad=False, device=Device.CUDA)
  print(a)

  b = Tensor(np.random.randn(100), requires_grad=False).to(Device.CUDA)
  print(b)

  c = a + b
  print(c.data)
  assert np.allclose(c.data, a.data + b.data), "CUDA addition failed"

  print("[+] Cuda test OK")
