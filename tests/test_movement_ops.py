#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd.tensor import Tensor
from picograd.util import is_cuda_available
from picograd.backend.device import Devices, Device

device = Device(Devices.CUDA) if is_cuda_available() else Device(Devices.CPU)
print("[*] Using device", device.name, "\n")

def test_reshape():
  # Basic reshape
  t = Tensor.random((2, 3), device=device)
  t_out = t.reshape(3, 2)
  assert t_out.shape == (3, 2), f"Expected shape (3, 2), got {t_out.shape}"
  t_out.backward()
  print("[+] Reshape Test OK")

  # Reshape to invalid shape
  try:
    t.reshape(4, 2)
  except ValueError:
    print("[+] Reshape Invalid Shape Test OK")

  # Reshape with zero dimensions
  t = Tensor.random((0, 3), device=device)
  t_out = t.reshape(3, 0)
  assert t_out.shape == (3, 0), f"Expected shape (3, 0), got {t_out.shape}"
  print("[+] Reshape Zero-Dimension Test OK")

def test_view():
  # Basic view
  t = Tensor.random((2, 3), device=device)
  t_out = t.view((3, 2))
  assert t_out.shape == (3, 2), f"Expected shape (3, 2), got {t_out.shape}"
  t_out.backward()
  print("[+] View Test OK")

  # View with zero dimensions
  t = Tensor.random((0, 3), device=device)
  t_out = t.view((3, 0))
  assert t_out.shape == (3, 0), f"Expected shape (3, 0), got {t_out.shape}"
  print("[+] View Zero-Dimension Test OK")

def test_transpose():
  # Basic transpose
  t = Tensor.random((2, 3), device=device)
  t_out = t.T
  assert t_out.shape == (3, 2), f"Expected shape (3, 2), got {t_out.shape}"
  t_out.backward()
  print("[+] Transpose Test OK")

  # Transpose of a square matrix
  t = Tensor.random((3, 3), device=device)
  t_out = t.T
  assert t_out.shape == (3, 3), f"Expected shape (3, 3), got {t_out.shape}"
  print("[+] Transpose Square Matrix Test OK")

  # Transpose of a 1D tensor
  t = Tensor.random((5,), device=device)
  t_out = t.T
  assert t_out.shape == (5,), f"Expected shape (5,), got {t_out.shape}"
  print("[+] Transpose 1D Tensor Test OK")

if __name__ == "__main__":
  test_reshape()
  test_view()
  test_transpose()


if __name__ == "__main__":
  test_reshape()
  test_view()
  test_transpose()
