#!/usr/bin/env python3
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd.tensor import Tensor
from picograd.draw_utils import draw_dot


if __name__ == "__main__":
  a = Tensor(np.array([1, 2, 3]))
  b = Tensor(np.array([4, 5, 6]))
  c = Tensor(np.array([7, 8, 9]))
  d = a * b + c
  d.backward()
  draw_dot(d)
  print("[+] DONE")