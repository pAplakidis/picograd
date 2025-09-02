#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import picograd
import picograd.nn as nn
from picograd.draw_utils import draw_dot

def test_batchnorm1d():
  t = picograd.Tensor.random((20, 5), name="input")
  bn = nn.BatchNorm1D(5)
  out = bn(t)
  draw_dot(out, path="graphs/batchnorm1d")
  out.backward()

if __name__ == "__main__":
  test_batchnorm1d()
