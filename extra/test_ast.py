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

# square arrays
# shape = (100, 100)
# a = Tensor.random(shape, lazy=lazy, device=device, name="a")
# b = Tensor.random(shape, lazy=lazy, device=device, name="b")
# c = Tensor.random(shape, lazy=lazy, device=device, name="c")

# non square arrays
a = Tensor.random((2, 4), lazy=lazy, device=device, name="a")
b = Tensor.random((4, 3), lazy=lazy, device=device, name="b")

# TODO: make this a unittest - cases for compiler:
# d = a * b + c
# d = a * b + c * d
# d = a * b + a
# d = a * b + a * c
# d = a * b + a * b + c
# d = a @ b + c
d = a @ b #+ c
# d = a @ b + a
# picograd.nn layers

# check = a.data * b.data + c.data
# check = a.data * b.data + c.data * d.data
# check = a.data * b.data + a.data
# check = a.data * b.data + a.data * c.data
# check = a.data * b.data + a.data * b.data + c.data
# check = a.data @ b.data + c.data
# check = a.data * b.data + a.data * b.data + c.data
# picograd.nn layers
check = a.data @ b.data # + c.data

# TODO: road to matmul
# unary ops: unsqueeze, expand, permute, reshape, view
# binary ops: elementwise add, mul
# reduce ops: sum (w/ axis)

d.realize()
assert np.allclose(check, d.data)
print("[+] OK")
