#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import numpy as np
from dataclasses import dataclass

# setup import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd.tensor import Tensor
from picograd.backend.device import Devices, Device
from picograd.backend.uop import UOp
from picograd.backend.dtypes import dtypes
from picograd.backend.function import OPS
from picograd.backend.linearizer import *
from picograd.backend.renderer.cuda import CUDARenderer

device = Device(Devices.CUDA)

lazy = True
a = Tensor.random((4, 4), lazy=lazy, device=device, name="a")
b = Tensor.random((4, 4), lazy=lazy, device=device, name="b")
c = Tensor.random((4, 4), lazy=lazy, device=device, name="c")

def ast_to_uops(ast_nodes):
  uop_map = {}
  uops = []
  for node in ast_nodes:
    if node.op is None:
      u = UOp(OPS.LOAD, dtypes.float32, arg=(node.tensor,))
    else:
      src = tuple(uop_map[inp] for inp in node.inputs)
      u = UOp(node.op, dtypes.float32, src=src)
    uop_map[node] = u
    uops.append(u)

  # FIXME: arg is not a tuple of tensors but various arguments for different ops such as dim, etc
  # final STORE
  uops.append(UOp(OPS.STORE, dtypes.float32, src=(uops[-1],), arg=(ast_nodes[-1].tensor,)))
  return uops

class ScheduleItem:
  pass

def create_schedule(ast):
  schedule = []
  return schedule

def run_schedule(schedule):
  for item in schedule:
    print(item)
    print()

# d = a @ b + c
d = a * b + c
# run_schedule(create_schedule(d.get_ast()))
uops = ast_to_uops(linearize(build_ast(d)))

for u in uops:
  print(u)
  print()

print(a.device_data, b.device_data, c.device_data)
renderer = CUDARenderer(arch="sm_80")
kernel_code, kernel_name = renderer.elementwise(OPS.ADD, dtypes.float32, (), shape=(a.shape))
kfunc = a.device.manager.compile_kernel(kernel_code, kernel_name.encode("utf-8"))

block_size = (256, 1, 1)
grid = ((np.prod(a.shape) + block_size[0] - 1) // block_size[0], 1, 1)
n_flops = np.prod(a.shape)

args = a.device.manager.prep_kargs(c.device_data, a.device_data, b.device_data)
a.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops=n_flops)

print(c.data)

# TODO: DEBUG
# opened device <DEVICE.name> from pid:<pid>
# *** <DEVICE.name>   <id> <data_type(empty, E_4_4=elementwise(4x4))>   <num_allocated_elemets(e.g. 16)> <dtype(e.g. dtype.float)>    arg <len(UOp.args)>   mem <allocated memory in GBs>
# if op, then add this at the end:
# <kernel_time_in_ms> <GFLOPs> <op_type(add, mul, dot, etc)>

# TODO: allocate CUDA memory for each tensor

# TODO: for each op, generate/render a CUDA kernel, compile and execute it

# TODO: test cases for compiler
# a * b + c
# a * b + c * d
# a * b + a
# a * b + a * c
# a @ b + c
# a @ b + a
# picograd.nn layers
