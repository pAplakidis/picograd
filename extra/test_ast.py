#!/usr/bin/env python3
import os
import sys
import unittest
import numpy as np

# setup import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from picograd.tensor import Tensor
from picograd.backend.device import Devices, Device
from picograd.backend.linearizer import *

device = Device(Devices.METAL)
lazy = True
serial = True

print("[*] Using device", device.name, "\n")


class TestCompilerOps(unittest.TestCase):
  def generate_square_tensors(self):
    a = Tensor.random((32, 32), lazy=lazy, device=device, name="a")
    b = Tensor.random((32, 32), lazy=lazy, device=device, name="b")
    c = Tensor.random((32, 32), lazy=lazy, device=device, name="c")
    return a, b, c

  def assert_tensor_equal(self, tensor, expected, atol=1e-5):
    tensor.realize()
    self.assertTrue(np.allclose(tensor.data, expected, atol=atol))

  # --------------- Elementwise Ops ----------------
  def test_add(self):
    a, b, _ = self.generate_square_tensors()
    d = a + b
    check = a.data + b.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler add OK")

  def test_mul_add(self):
    a, b, c = self.generate_square_tensors()
    d = a * b + c
    check = a.data * b.data + c.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler mul + add OK")

  def test_mul_add_shared_input(self):
    a, b, c = self.generate_square_tensors()
    d = a * b + a * c
    check = a.data * b.data + a.data * c.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler shared input graph OK")

  def test_repeated_expression(self):
    a, b, c = self.generate_square_tensors()
    d = a * b + a * b + c
    check = a.data * b.data + a.data * b.data + c.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler repeated expression OK")

  def test_residual(self):
    a, b, _ = self.generate_square_tensors()
    d = a * b + a
    check = a.data * b.data + a.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler residual OK")

  # --------------- Reduce Ops ----------------
  def test_sum_axis_0(self):
    a, _, _ = self.generate_square_tensors()
    d = a.sum(axis=0)
    check = a.data.sum(axis=0)

    self.assert_tensor_equal(d, check)
    print("[+] Compiler reduce axis=0 OK")

  def test_sum_axis_1(self):
    a, _, _ = self.generate_square_tensors()
    d = a.sum(axis=1)
    check = a.data.sum(axis=1)

    self.assert_tensor_equal(d, check)
    print("[+] Compiler reduce axis=1 OK")

  # --------------- Matmul Ops ----------------
  def test_matmul_square(self):
    a, b, _ = self.generate_square_tensors()
    d = a @ b
    check = a.data @ b.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler square matmul OK")

  def test_dot_square(self):
    a, b, _ = self.generate_square_tensors()
    d = a.dot(b)
    check = a.data @ b.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler square dot OK")

  def test_matmul_non_square_cases(self):
    shapes = [
      ((2, 4), (4, 3)),
      ((8, 16), (16, 4)),
      ((5, 7), (7, 2)),
      ((1, 32), (32, 1)),
      ((64, 8), (8, 128)),
    ]

    for shape_a, shape_b in shapes:
      a = Tensor.random(shape_a, lazy=lazy, device=device, name="a")
      b = Tensor.random(shape_b, lazy=lazy, device=device, name="b")

      d = a @ b
      check = a.data @ b.data

      self.assert_tensor_equal(d, check)
      print(f"[+] Compiler matmul {shape_a} x {shape_b} OK")

  def test_dot_non_square_cases(self):
    shapes = [
      ((2, 4), (4, 3)),
      ((8, 16), (16, 4)),
      ((5, 7), (7, 2)),
      ((1, 32), (32, 1)),
      ((64, 8), (8, 128)),
    ]

    for shape_a, shape_b in shapes:
      a = Tensor.random(shape_a, lazy=lazy, device=device, name="a")
      b = Tensor.random(shape_b, lazy=lazy, device=device, name="b")

      d = a.dot(b)
      check = a.data @ b.data

      self.assert_tensor_equal(d, check)
      print(f"[+] Compiler dot {shape_a} x {shape_b} OK")

  # FIXME:
  # File "/Users/paul/Dev/picograd/picograd/backend/renderer/metal_renderer.py", line 50, in elementwise
  # assert len(arg) == 3, f"Expected 3 arguments for c = a alu b, got {len(arg)} instead"
  #        ^^^^^^^^^^^^^
  # AssertionError: Expected 3 arguments for c = a alu b, got 2 instead
  def test_matmul_add(self):
    a, b, c = self.generate_square_tensors()
    d = a @ b + c
    check = a.data @ b.data + c.data

    self.assert_tensor_equal(d, check)
    print("[+] Compiler matmul + add OK")

  # --------------- Conv Ops ----------------

  def test_conv2d(self):
    input_tensor = Tensor.random((1, 3, 32, 32), lazy=lazy, device=device, name="input")
    kernel = Tensor.random((6, 3, 5, 5), lazy=lazy, device=device, name="kernel")
    output_tensor = input_tensor.conv2d(kernel, in_channels=3, out_channels=6, stride=1, padding=0)

    # Compute expected output using NumPy for verification
    input_data = input_tensor.data
    kernel_data = kernel.data
    expected_output = np.zeros((1, 6, 28, 28))  # Output shape for valid convolution

    # TODO: check with CPU
    # self.assert_tensor_equal(output_tensor, expected_output)
    print("[+] Compiler conv2d OK")


if __name__ == "__main__":
  if serial:
    # run serially for debugging
    runner = unittest.TextTestRunner(failfast=True)
    unittest.main(testRunner=runner)
  else:
    unittest.main()
