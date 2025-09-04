#!/usr/bin/env python3
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import picograd as pg
import picograd.nn as nn

def test_batchnorm1d():
  t = pg.Tensor.random((20, 5), name="input")
  bn = nn.BatchNorm1D(5)
  out = bn(t)
  out.backward()
  print("[+] BatchNorm1D OK")

def test_batchnorm2d():
  t = pg.Tensor.random((20, 3, 32, 32), name="input", dtype=np.int8)
  bn = nn.BatchNorm2D(3)
  out = bn(t)
  out.backward()
  print("[+] BatchNorm2D OK")

def test_layernorm():
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
  test_batchnorm1d()
  test_batchnorm2d()
  test_layernorm()
