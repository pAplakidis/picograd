import os
import numpy as np

from picograd.tensor import Tensor
from picograd.print_utils import *
from picograd.backend.uop import UOp
from picograd.backend.dtypes import dtypes
from picograd.backend.function import OPS
from picograd.backend.linearizer import *


DEBUG = int(os.getenv("DEBUG", 0))


class ScheduleItem:
  pass


class Scheduler:
  def __init__(self, ast, renderer):
    self.ast = ast
    self.renderer = renderer
    self.mngr = ast[0].tensor.device.manager

  @staticmethod
  def ast_to_uops(ast_nodes):
    uop_map = {}
    uops = []
    for node in ast_nodes:
      if node.op is None:
        u = UOp(OPS.LOAD, dtypes.float32, arg=(node.tensor,))
      else:
        src = tuple(uop_map[inp] for inp in node.inputs)
        u = UOp(node.op, dtypes.float32, src=src, arg=(node.tensor,))
      uop_map[node] = u
      uops.append(u)

    # TODO: arg is not just a tuple of tensors but various arguments for different ops such as dim, etc
    # final STORE
    uops.append(UOp(OPS.STORE, dtypes.float32, src=(uops[-1],), arg=(ast_nodes[-1].tensor,)))
    return uops

  def create_schedule(self):
    self.schedule = Scheduler.ast_to_uops(self.ast)
    return self.schedule

  def run_schedule(self):
    for id, item in enumerate(self.schedule):
      if DEBUG >= 2:
        print(item)
      self.lower_item(item, id)

  def lower_item(self, item: UOp, id: int = 0):
    if DEBUG >= 1 and id == 0: print(f"<DEVICE> <ID> <KERNEL_NAME> <NUM_ELEMS> <DTYPE>    <NUM_ARGS>   <MEMORY_IN_GB> <kernel_time> - <GFLOPs> <op_type>")

    # TODO: if LOAD, move data to device (if not already) - will be different later when tensors aren't allcoated in __init__()
    # if item.op == OPS.LOAD:
    #   tensor = item.arg[0]
    #   tensor.device_data = tensor.device.manager.to_device(tensor.data)
    #   return

    if item.op in (OPS.ADD, OPS.MUL):
      # TODO: make args more generic (cover all ops)
      args = [uop.arg[0] for uop in item.src if uop.op in (OPS.LOAD, OPS.ADD, OPS.MUL)] # input tensors
      args.insert(0, item.arg[0])  # output tensor
      kernel_code, kernel_name = self.renderer.elementwise(item.op, dtypes.float32, (), shape=(args[0].shape))
      if DEBUG >= 2: print(kernel_code)

      kfunc = self.mngr.compile_kernel(kernel_code, kernel_name.encode("utf-8"))
      elapsed_ms, gflops = self.run_kernel(kfunc, args, shape=args[0].shape)
      if DEBUG >= 1:
        # TODO: don't use tensor._data (tensor might be 100% on the device)
        debug_str = f"{color_green(f'*** {self.mngr.dev_name} {id}')} {color_red(kernel_name)}  {len(item.src) } {item.dtype.name}   arg {len(item.arg) if item.arg else 0}   mem {sum(tensor._data.nbytes for uop in item.src for tensor in uop.arg) / (1024**3):.6f} GB"
        if item.op in (OPS.ADD, OPS.MUL):
          debug_str += f"   {elapsed_ms:.4f} ms - {gflops:.4f} GFLOPs)   {item.op.name.lower()}"
        print(debug_str)

  def run_kernel(self, kfunc, args: list, shape: tuple):
    kargs = self.mngr.prep_kargs(*[arg.device_data if isinstance(arg, Tensor) else arg for arg in args])
    block_size = (1, 1, 1)
    grid = (np.prod(shape), 1, 1)
    n_flops = np.prod(shape)
    return self.mngr.launch_kernel(kfunc, grid, block_size, kargs, n_flops=n_flops)
