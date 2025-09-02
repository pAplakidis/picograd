#!/usr/bin/env python3
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import picograd
import picograd.nn as nn
from picograd.draw_utils import draw_dot

def test_batchnorm1d():
  t = picograd.Tensor.random((20, 5), name="input")
  bn = nn.BatchNorm1D(5)
  out = bn(t)
  out.backward()
  print("[+] BatchNorm1D OK")

def test_batchnorm2d():
  t = picograd.Tensor.random((20, 3, 32, 32), name="input", dtype=np.int8)
  bn = nn.BatchNorm2D(3)
  out = bn(t)
  out.backward()
  print("[+] BatchNorm2D OK")

if __name__ == "__main__":
  test_batchnorm1d()
  test_batchnorm2d()
