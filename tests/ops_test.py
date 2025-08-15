#!/usr/bin/env python3
import unittest
from time import time
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd.tensor import Tensor
from picograd.util import is_cuda_available
from picograd.backend.device import Devices, Device
from picograd.draw_utils import draw_dot


cpu = Device(Devices.CPU)
# device = Device(Devices.CPU)
device = Device(Devices.CUDA) if is_cuda_available() else Device(Devices.CPU)
print("[*] Using device", device.name, "\n")


# TODO: move tensor to CPU at the end of each test
class TestOps(unittest.TestCase):
  def test_add(self):
    try:
      vec1 = Tensor(np.random.randn(10000), device=device)
      vec2 = Tensor(np.random.randn(10000), device=device)
      vec3 = vec1 + vec2
      vec3.backward()
      assert np.allclose(vec3.data, vec1.data + vec2.data), "vector addition failed"

      mat1 = Tensor(np.random.randn(512, 512), device=device)
      mat2 = Tensor(np.random.randn(512, 512), device=device)
      mat3 = mat1 + mat2
      mat3.backward()
      assert np.allclose(mat3.data, mat1.data + mat2.data), "matrix addition failed"

      tensor1 = Tensor(np.random.randn(64, 64, 64), device=device)
      tensor2 = Tensor(np.random.randn(64, 64, 64), device=device)
      tensor3 = tensor1 + tensor2
      tensor3.backward()
      assert np.allclose(tensor3.data, tensor1.data + tensor2.data), "tensor addition failed"

      print("[+] Addition test passed\n")
    except Exception as e:
      self.fail(f"[!] Addition test failed: {e}\n")

  def test_mul(self):
    try:
      vec1 = Tensor(np.random.randn(10000), device=device)
      vec2 = Tensor(np.random.randn(10000), device=device)
      vec3 = vec1 * vec2
      vec3.backward()
      assert np.allclose(vec3.data, vec1.data * vec2.data), "vector elementwise multiplication failed"

      mat1 = Tensor(np.random.randn(512, 512), device=device)
      mat2 = Tensor(np.random.randn(512, 512), device=device)
      mat3 = mat1 * mat2
      mat3.backward()
      assert np.allclose(mat3.data, mat1.data * mat2.data), "matrix elementwise multiplication failed"

      tensor1 = Tensor(np.random.randn(64, 64, 64), device=device)
      tensor2 = Tensor(np.random.randn(64, 64, 64), device=device)
      tensor3 = tensor1 * tensor2
      tensor3.backward()
      assert np.allclose(tensor3.data, tensor1.data * tensor2.data), "tensor elementwise multiplication failed"

      print("[+] Elementwise multiplication test passed\n")
    except Exception as e:
      self.fail(f"[!] Elementwise multiplication test failed: {e}\n")

  def test_linear_layer(self):
    try:
      t1 = np.random.randn(100, 50).astype(np.float32)
      t2 = np.random.randn(50, 100).astype(np.float32)
      t3 = np.random.randn(100, 100).astype(np.float32)

      a = Tensor(t1, name="a", device=device)
      b = Tensor(t2, name="b", device=device)

      c = a.dot(b)
      c.name = "c"
      assert np.allclose(c.data, a.data @ b.data, atol=1e-4), "dot product failed"

      d = Tensor(t3, name="d", device=device)
      print(c.name, c.device, c.device_data)
      e = c + d
      print(c.name, c.device_grad)
      e.name = "e"
      e.backward()
      assert np.allclose(e.data, c.data + d.data), "addition failed"

      draw_dot(e, path="graphs/ops_test")

      a_cpu = Tensor(t1, device=cpu)
      b_cpu = Tensor(t2, device=cpu)
      c_cpu = a_cpu.dot(b_cpu)
      assert np.allclose(c_cpu.data, c.data, atol=1e-4), "dot product data mismatch between CPU and CUDA dot product"

      d_cpu = Tensor(t3, device=cpu)
      e_cpu = c_cpu + d_cpu
      e_cpu.backward()
      assert np.allclose(e_cpu.data, e.data, atol=1e-4), "addition data mismatch between CPU and CUDA addition"

      print("E")
      print(e.grad)
      print(e_cpu.grad)
      print()
      assert np.allclose(e.grad, e_cpu.grad, atol=1e-4), "gradient of a mismatch"
      print("D")
      print(d.grad)
      print(d_cpu.grad)
      print()
      # FIXME: running this separetely works, but not in the whole test suite (d.grad in CUDA is != 0.0)
      assert np.allclose(d.grad, d_cpu.grad, atol=1e-4), "gradient of a mismatch"
      print("C")
      print(c.grad)
      print(c_cpu.grad)
      print()
      assert np.allclose(c.grad, c_cpu.grad, atol=1e-4), "gradient of a mismatch"
      print("B")
      print(b.grad)
      print(b_cpu.grad)
      print()
      assert np.allclose(b.grad, b_cpu.grad, atol=1e-4), "gradient of a mismatch"
      print("A")
      print(a.grad)
      print(a_cpu.grad)
      print()
      assert np.allclose(a.grad, a_cpu.grad, atol=1e-4), "gradient of a mismatch"

      print("[+] Linear layer op OK\n")
    except Exception as e:
      self.fail(f"[!] Linear layer test failed: {e}\n")

  def test_conv2d_layer(self):
    try:
      a = Tensor(np.random.randn(1, 3, 32, 32), device=device)  # (batch_size, channels, height, width)
      w = Tensor(np.random.randn(16, 3, 3, 3), device=device)  # (out_channels, kernel_height, kernel_width)
      b = Tensor(np.zeros((16,)), device=device)  # (out_channels,)

      c = a.conv2d(w, b, 3, 16)
      c.backward()
      # TODO: assert results
      print("[+] Conv2D OK\n")

    except Exception as e:
      self.fail(f"[!] Conv2D layer test failed: {e}\n")


if __name__ == "__main__":
  unittest.main()
  print("[+] Cuda test OK")
