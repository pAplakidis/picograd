#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import numpy as np

# setup import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd.tensor import Tensor
from picograd.backend.device import Devices, Device
from picograd.backend.linearizer import *

device = Device(Devices.CUDA)
lazy = True
shape = (100, 100)
a = Tensor.random(shape, lazy=lazy, device=device, name="a")
b = Tensor.random(shape, lazy=lazy, device=device, name="b")
c = Tensor.random(shape, lazy=lazy, device=device, name="c")

d = a * b + a * b + c
# TODO: test cases for compiler
# a * b + c
# a * b + c * d
# a * b + a
# a * b + a * c
# a @ b + c
# a @ b + a
# picograd.nn layers

# TODO: road to matmul
# unary ops: unsqueeze, expand, permute, reshape, view
# binary ops: elementwise add, mul
# reduce ops: sum (w/ axis)

d.realize()
check = a.data * b.data + a.data * b.data + c.data
assert np.allclose(check, d.data)
print("[+] OK")
