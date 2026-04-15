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

device = Device(Devices.METAL)
lazy = True

# square arrays
# FIXME: matmul breaks with large shapes (>= 32) in metal, max tid=1024, 32*32*32=32768 > 1024, need to implement tiling
# shape = (32, 32)
# a = Tensor.random(shape, lazy=lazy, device=device, name="a")
# b = Tensor.random(shape, lazy=lazy, device=device, name="b")
# c = Tensor.random(shape, lazy=lazy, device=device, name="c")

# non square arrays
a = Tensor.random((2, 4), lazy=lazy, device=device, name="a")
b = Tensor.random((4, 3), lazy=lazy, device=device, name="b")

# TODO: make unittests - cases for compiler:

# ELEMENTWISE OPS
# d = a + b
# check = a.data + b.data

# d = a * b + c
# check = a.data * b.data + c.data

# d = a * b + c * d
# check = a.data * b.data + c.data * d.data

# d = a * b + a
# check = a.data * b.data + a.data

# d = a * b + a * c
# check = a.data * b.data + a.data * c.data

# d = a * b + a * b + c
# check = a.data * b.data + a.data * b.data + c.data

# ------------------

# REDUCE OPS

# d = a.sum(axis=0)
# check = a.data.sum(axis=0)

# ------------------

# MIXED OPS

# NOTE: subops to implement matmul:
# unary ops: unsqueeze, expand, permute, reshape, view
# binary ops: elementwise add, mul
# reduce ops: sum (w/ axis)

d = a @ b
check = a.data @ b.data

# d = a @ b + c
# check = a.data @ b.data + c.data
# check = a.data * b.data + a.data * b.data + c.data

# d = a @ b + a
# ------------------

# TODO: picograd.nn layers


d.realize()
assert np.allclose(check, d.data)
print("[+] OK")
