#!/usr/bin/env python3
import os
import sys
import unittest
import numpy as np

# setup import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import picograd as pg
import picograd.nn as nn
from picograd.tensor import Tensor
from picograd.util import is_cuda_available
from picograd.backend.device import Devices, Device

cpu = Device(Devices.CPU)
# device = Device(Devices.CUDA) if is_cuda_available() else Device(Devices.CPU)
device = cpu
print("[*] Using device", device.name, "\n")


# --------------- Binary Ops ----------------
# TODO: more binary ops (pow, sub, div, etc)
class TestBinaryOps(unittest.TestCase):
  def generate_tensors(self):
    t1 = np.random.randn(100, 100).astype(np.float32)
    t2 = np.random.randn(100, 100).astype(np.float32)
    a = Tensor(t1, name="a", device=device)
    b = Tensor(t2, name="b", device=device)
    return a, b

  def test_add(self):
    vec1 = Tensor.random((10000,), device=device)
    vec2 = Tensor.random((10000,), device=device)
    vec3 = vec1 + vec2
    vec3.backward()
    self.assertTrue(np.allclose(vec3.data, vec1.data + vec2.data))

    mat1 = Tensor.random((512, 512), device=device)
    mat2 = Tensor.random((512, 512), device=device)
    mat3 = mat1 + mat2
    mat3.backward()
    self.assertTrue(np.allclose(mat3.data, mat1.data + mat2.data))

    tensor1 = Tensor(np.random.randn(64, 64, 64), device=device)
    tensor2 = Tensor(np.random.randn(64, 64, 64), device=device)
    tensor3 = tensor1 + tensor2
    tensor3.backward()
    self.assertTrue(np.allclose(tensor3.data, tensor1.data + tensor2.data))
    print("[+] Addition test passed")

  def test_mul(self):
    vec1 = Tensor.random((10000,), device=device)
    vec2 = Tensor.random((10000,), device=device)
    vec3 = vec1 * vec2
    vec3.backward()
    self.assertTrue(np.allclose(vec3.data, vec1.data * vec2.data))

    mat1 = Tensor.random((512, 512), device=device)
    mat2 = Tensor.random((512, 512), device=device)
    mat3 = mat1 * mat2
    mat3.backward()
    self.assertTrue(np.allclose(mat3.data, mat1.data * mat2.data))

    tensor1 = Tensor.random((64, 64, 64), device=device)
    tensor2 = Tensor.random((64, 64, 64), device=device)
    tensor3 = tensor1 * tensor2
    tensor3.backward()
    self.assertTrue(np.allclose(tensor3.data, tensor1.data * tensor2.data))
    print("[+] Multiplication test passed")

  def test_linear_layer(self):
    t1 = np.random.randn(100, 50).astype(np.float32)
    t2 = np.random.randn(50, 100).astype(np.float32)
    t3 = np.random.randn(100, 100).astype(np.float32)

    a = Tensor(t1, name="a", device=device)
    b = Tensor(t2, name="b", device=device)

    c = a.dot(b)
    c.name = "c"
    self.assertTrue(np.allclose(c.data, a.data @ b.data, atol=1e-4))

    d = Tensor(t3, name="d", device=device)
    e = c + d
    e.name = "e"
    e.backward()
    self.assertTrue(np.allclose(e.data, c.data + d.data))

    # Compare CPU vs device
    a_cpu = Tensor(t1, device=cpu)
    b_cpu = Tensor(t2, device=cpu)
    c_cpu = a_cpu.dot(b_cpu)
    self.assertTrue(np.allclose(c_cpu.data, c.data, atol=1e-4))

    d_cpu = Tensor(t3, device=cpu)
    e_cpu = c_cpu + d_cpu
    e_cpu.backward()
    self.assertTrue(np.allclose(e_cpu.data, e.data, atol=1e-4))
    print("[+] Linear layer op OK")

  def test_conv2d_layer(self):
    a = Tensor(np.random.randn(1, 3, 32, 32), device=device)
    w = Tensor(np.random.randn(16, 3, 3, 3), device=device)
    b = Tensor(np.zeros((16,)), device=device)

    c = a.conv2d(w, b, 3, 16)
    c.backward()
    # TODO: assert results
    print("[+] Conv2D OK")

  def test_residual(self):
    a, b, = self.generate_tensors()
    c = (a @ b) + a
    c.backward()
    print("[+] Residual OK")


# --------------- Unary Ops ----------------
class TestUnaryOps(unittest.TestCase):
  def generate_tensors(self):
    t1 = np.random.randn(100, 50).astype(np.float32)
    a = Tensor(t1, name="a", device=device)
    b = Tensor(t1, name="b", device=cpu)
    return a, b, t1

  def test_relu(self):
      a, _, t1 = self.generate_tensors()
      res = a.relu()
      res.backward()
      self.assertTrue(np.allclose(res.data, np.maximum(0, t1)))
      print("[+] ReLU OK")

  def test_softmax(self):
      a, b, _ = self.generate_tensors()
      res_a = a.softmax()
      res_a.backward()
      self.assertTrue(np.allclose(res_a.data, b.softmax().data))
      print("[+] Softmax OK")

  def test_tanh(self):
      a, _, _ = self.generate_tensors()
      res = a.tanh()
      res.backward()
      print("[+] Tanh OK")

  def test_sigmoid(self):
      a, _, _ = self.generate_tensors()
      res = a.sigmoid()
      res.backward()
      print("[+] Sigmoid OK")

# --------------- Movement Ops ----------------
class TestMovementOps(unittest.TestCase):
  def test_reshape(self):
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

  def test_view(self):
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

  def test_transpose(self):
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


# --------------- Movement Ops ----------------
class TestReduceOps(unittest.TestCase):
  def test_reduce_ops(self):
    a = Tensor.random((50, 100), name="a", device=device)
    a.max()
    a.min()
    a.sum()
    a.mean()
    a.std()
    a.argmax()
    a.argmin()

    a.max(axis=1)
    a.min(axis=1)
    a.sum(axis=1)
    a.mean(axis=1)
    a.std(axis=1)
    a.argmax(axis=1)
    a.argmin(axis=1)
    print("[+] Reduce Ops OK")


# --------------- Pooling ----------------
class TestPooling(unittest.TestCase):
  def test_maxpool2d(self):
    t1 = np.random.randn(1, 3, 10, 10).astype(np.float32)
    a = Tensor(t1, name="a", device=device)
    # torch_a = torch.tensor(t1)
    res = a.maxpool2d()
    res.backward()
    # torch_res = nn.MaxPool2d(kernel_size=2, stride=1)(torch_a).numpy()
    # assert np.allclose(res.data, torch_res), "MaxPool2d output does not match PyTorch output"
    print("[+] MaxPool2D OK")

  def test_avgpool2d(self):
    t1 = np.random.randn(1, 3, 10, 10).astype(np.float32)
    a = Tensor(t1, name="a", device=device)
    # torch_a = torch.tensor(t1)
    res = a.avgpool2d()
    res.backward()
    # torch_res = nn.AvgPool2d(kernel_size=2, stride=1)(torch_a).numpy()
    # assert np.allclose(res.data, torch_res), "AvgPool2d output does not match PyTorch output"
    print("[+] AvgPool2D OK")


# --------------- Norm ----------------
class TestNorm(unittest.TestCase):
  def test_batchnorm1d(self):
    t = pg.Tensor.random((20, 5), name="input")
    bn = nn.BatchNorm1D(5)
    out = bn(t)
    out.backward()
    print("[+] BatchNorm1D OK")

  def test_batchnorm2d(self):
    t = pg.Tensor.random((20, 3, 32, 32), name="input", dtype=np.int8)
    bn = nn.BatchNorm2D(3)
    out = bn(t)
    out.backward()
    print("[+] BatchNorm2D OK")

  def test_layernorm(self):
    t = pg.Tensor.random((32, 128), name="input")
    ln = nn.LayerNorm(128)
    out = ln(t)
    out.backward()
    print("[+] LayerNorm for MLP input OK")

    B, C, H, W = 16, 64, 7, 7
    t = pg.Tensor.random((B, C, H, W), name="input")
    ln = nn.LayerNorm((C, H, W))
    out = ln(t)
    out.backward()
    print("[+] LayerNorm for conv2D input OK")


if __name__ == "__main__":
  unittest.main()
