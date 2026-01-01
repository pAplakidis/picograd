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
from picograd.backend.scheduler import Scheduler
from picograd.backend.renderer.cuda_renderer import CUDARenderer

device = Device(Devices.CUDA)
lazy = True
a = Tensor.random((4, 4), lazy=lazy, device=device, name="a")
b = Tensor.random((4, 4), lazy=lazy, device=device, name="b")
c = Tensor.random((4, 4), lazy=lazy, device=device, name="c")

d = a * b + c
# TODO: test cases for compiler
# a * b + c
# a * b + c * d
# a * b + a
# a * b + a * c
# a @ b + c
# a @ b + a
# picograd.nn layers

# TODO: move to tensor.py
def realize(t: Tensor):
  renderer = CUDARenderer(arch="sm_80")
  scheduler = Scheduler(linearize(build_ast(t)), renderer)
  scheduler.create_schedule()
  scheduler.run_schedule()
  print(a.device_data, b.device_data, c.device_data)

# FIXME: the outputs must match (we don't use size anymore, so block_size and grid must be derived from tensor sizes)
realize(d)
print("expected:", a.data * b.data + c.data)
print("got:", d.data)
assert np.allclose(a.data * b.data + c.data, d.data)
