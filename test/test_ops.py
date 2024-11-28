#!/usr/bin/env python3
import unittest
import torch
import numpy as np

# TODO: find a way to import without this
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd.tensor import Tensor

class TestOps(unittest.TestCase):
  def test_tensor_ops(self):
    try:
      a = Tensor(np.array(-4.0))
      b = Tensor(np.array(2.0))
      c = a + b
      d = a * b + b**3
      c += c + Tensor(np.array(1.0))
      c += Tensor(np.array(1.0)) + c + (-a)
      d += d * Tensor(np.array(2.0)) + (b + a).relu()
      d += Tensor(np.array(3.0)) * d + (b - a).relu()
      e = c - d
      f = e**2
      g = f / Tensor(np.array(2.0))
      g += Tensor(np.array(10.0)) / f
      g.backward()
      amg, bmg, gmg = a, b, g

      a = torch.Tensor([-4.0]).double()
      b = torch.Tensor([2.0]).double()
      a.requires_grad = True
      b.requires_grad = True
      c = a + b
      d = a * b + b**3
      c = c + c + torch.Tensor([1.0])
      c = c + 1 + c + (-a)
      d = d + d * 2 + (b + a).relu()
      d = d + 3 * d + (b - a).relu()
      e = c - d
      f = e**2
      g = f / 2.0
      g = g + 10.0 / f
      g.backward()
      apt, bpt, gpt = a, b, g

      tol = 1e-6
      # check forward
      assert abs(gmg.data - gpt.data.item()) < tol  # FIXME: fails
      # check backward
      assert abs(amg.grad - apt.grad.item()) < tol
      assert abs(bmg.grad - bpt.grad.item()) < tol

      self.assertTrue(True)
    except Exception as e:
      self.fail(f"Test failed: {e}")


if __name__ == "__main__":
  unittest.main()
