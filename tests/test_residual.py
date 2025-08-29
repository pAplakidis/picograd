#!/usr/bin/env python3
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd.tensor import Tensor
from picograd.backend.device import Devices, Device
from picograd.draw_utils import draw_dot

device = Device(Devices.CPU)
print("[*] Using device", device.name, "\n")

def generate_tensors():
  t1 = np.random.randn(100, 100).astype(np.float32)
  t2 = np.random.randn(100, 100).astype(np.float32)
  a = Tensor(t1, name="a", device=device)
  b = Tensor(t2, name="b", device=Device(Devices.CPU))
  return a, b

def test_residual():
  a, b, = generate_tensors()
  c = (a @ b) + a
  c.backward()
  draw_dot(c, path="graphs/restest")

  print("[+] Residual OK")


if __name__ == "__main__":
  test_residual()
