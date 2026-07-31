import os
import numpy as np

from picograd.print_utils import *
from picograd.backend.uop import UOp
from picograd.backend.dtypes import dtypes
from picograd.backend.function import OPS, BINARY_OPS,  MOVEMENT_OPS, REDUCE_OPS
from picograd.backend.linearizer import *
from picograd.viz import recorder as viz


DEBUG = int(os.getenv("DEBUG", 0))


def _uop_summary(item: UOp):
  arg = item.arg
  arg_len = len(arg) if isinstance(arg, (tuple, list)) else (0 if arg is None else 1)
  return {
    "op": item.op.name if hasattr(item.op, "name") else str(item.op),
    "dtype": item.dtype.name if hasattr(item.dtype, "name") else str(item.dtype),
    "src_count": len(item.src) if item.src else 0,
    "arg_count": arg_len,
    "tag": item.tag,
  }


def _uop_repr(item: UOp, limit: int = 2000):
  text = repr(item)
  return text if len(text) <= limit else text[:limit] + "..."


# TODO: check out [ https://mesozoic-egg.github.io/tinygrad-notes/scheduleitem.html ]
# TODO: ScheduleItem => Linear Representation (UOps) [ https://mesozoic-egg.github.io/tinygrad-notes/uops.html ]
class ScheduleItem:
  pass


class Scheduler:
  def __init__(self, ast, renderer):
    self.ast = ast
    self.renderer = renderer
    self.mngr = ast[0].tensor.device.manager

  def debug_prefix(self, id):
    return color_green(f'*** {self.mngr.dev_name} {id}')

  def device_name(self):
    return getattr(self.mngr.dev_name, "name", str(self.mngr.dev_name))

  @staticmethod
  def ast_to_uops(ast_nodes):
    uop_map = {}
    uops = []
    for node in ast_nodes:
      if node.op is None:
        u = UOp(OPS.LOAD, dtypes.float32, arg=(node.tensor,))
      else:
        src = tuple(uop_map[inp] for inp in node.inputs)
        u = UOp(node.op, dtypes.float32, src=src, arg=(node.tensor, node.forward_args, node.forward_kwargs))
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
        print()
      self.lower_item(item, id)

  # TODO: proper UOps to CStyle [ https://mesozoic-egg.github.io/tinygrad-notes/backends.html ]
  def lower_item(self, item: UOp, id: int = 0):
    if DEBUG >= 1 and id == 0: print(f"<DEVICE> <ID> <KERNEL_NAME> <NUM_ELEMS> <DTYPE>    <NUM_ARGS>   <MEMORY_IN_GB> <kernel_time> - <GFLOPs> <op_type>")
    viz.record("uop", schedule_id=id, op=item.op, dtype=item.dtype, src=item.src, arg=item.arg, summary=_uop_summary(item), repr=_uop_repr(item))
    # TODO: use pattern_matcher => self.renderer.render_code(item)

    # TODO: if LOAD, move data to device (if not already) - will be different later when tensors aren't allcoated in __init__()
    if item.op == OPS.LOAD:
      tensor = item.arg[0]
      # tensor.device_data = tensor.device.manager.to_device(tensor.data)
      if viz.enabled(): viz.record("schedule_copy", schedule_id=id, direction="load", tensor=tensor, bytes=tensor.nbytes, device=self.device_name())
      if DEBUG >= 1: print(f"{self.debug_prefix(id)} {color_yellow('copy')}  {len(item.src) } {item.dtype.name}   arg {len(item.arg) if item.arg else 0}   mem {tensor._data.nbytes / (1024**3):.6f} GB")
      return

    if item.op == OPS.STORE:
      tensor = item.arg[0]
      # tensor.data = tensor.device.manager.to_host(tensor.device_data)
      if viz.enabled(): viz.record("schedule_copy", schedule_id=id, direction="store", tensor=tensor, bytes=tensor.nbytes, device=self.device_name())
      if DEBUG >= 1: print(f"{self.debug_prefix(id)} {color_yellow('copy')}  {len(item.src) } {item.dtype.name}   arg {len(item.arg) if item.arg else 0}   mem {tensor._data.nbytes / (1024**3):.6f} GB")
      return

    if item.op in MOVEMENT_OPS and DEBUG >= 1:
      print(f"{self.debug_prefix(id)} {color_yellow('copy')}  {len(item.src) } {item.dtype.name}   arg {len(item.arg) if item.arg else 0}   mem {sum(uop.arg[0]._data.nbytes for uop in item.src) / (1024**3):.6f} GB {item.op.name.lower()}")

    if item.op in (OPS.ADD, OPS.MUL):
      args = [item.arg[0], *(src.arg[0] for src in item.src)]
      kernel_code, kernel_name, contiguous = self.renderer.elementwise(item.op, dtypes.float32, args)
      if DEBUG >= 2: print('\n', kernel_code, '\n')

      if self.mngr.dev_name == "CUDA": kernel_name = kernel_name.encode("utf-8")
      kfunc = self.mngr.compile_kernel(kernel_code, kernel_name)
      elapsed_ms, gflops = self.run_elementwise_kernel(kfunc, args, shape=args[0].shape, contiguous=contiguous)
      if viz.enabled():
        grid, block = self.launch_dims(int(np.prod(args[0].shape)))
        viz.record_kernel(name=kernel_name, source=kernel_code, device=self.device_name(), schedule_id=id, uop_summary=_uop_summary(item), uop_repr=_uop_repr(item), op=item.op, args=args, shape=args[0].shape, grid=grid, block=block, contiguous=contiguous, memory_bytes=sum(uop.arg[0].nbytes for uop in item.src), elapsed_ms=elapsed_ms, gflops=gflops)
      if DEBUG >= 1:
        # TODO: don't use tensor._data (tensor might be 100% on the device)
        debug_str = f"{self.debug_prefix(id)} {color_red(kernel_name)}  {len(item.src) } {item.dtype.name}   arg {len(item.arg) if item.arg else 0}   mem {sum(uop.arg[0]._data.nbytes for uop in item.src) / (1024**3):.6f} GB"
        if item.op in (OPS.ADD, OPS.MUL):
          debug_str += f"   ({elapsed_ms:.4f} ms - {gflops:.4f} GFLOPs)   {item.op.name.lower()}"
        print(debug_str)
      return

    if item.op == OPS.ReLU:
      args = [item.arg[0], item.src[0].arg[0]]
      kernel_code, kernel_name = self.renderer.relu(dtypes.float32, args)
      if DEBUG >= 2: print('\n', kernel_code, '\n')

      if self.mngr.dev_name == "CUDA": kernel_name = kernel_name.encode("utf-8")
      kfunc = self.mngr.compile_kernel(kernel_code, kernel_name)
      elapsed_ms, gflops = self.run_unary_kernel(kfunc, args, shape=args[0].shape)
      if viz.enabled():
        grid, block = self.launch_dims(int(np.prod(args[0].shape)))
        viz.record_kernel(name=kernel_name, source=kernel_code, device=self.device_name(), schedule_id=id, uop_summary=_uop_summary(item), uop_repr=_uop_repr(item), op=item.op, args=args, shape=args[0].shape, grid=grid, block=block, memory_bytes=sum(uop.arg[0].nbytes for uop in item.src), elapsed_ms=elapsed_ms, gflops=gflops)
      if DEBUG >= 1:
        debug_str = f"{self.debug_prefix(id)} {color_red(kernel_name)}  {len(item.src) } {item.dtype.name}   arg {len(item.arg) if item.arg else 0}   mem {sum(uop.arg[0]._data.nbytes for uop in item.src) / (1024**3):.6f} GB"
        debug_str += f"   ({elapsed_ms:.4f} ms - {gflops:.4f} GFLOPs)   relu"
        print(debug_str)
      return

    if item.op == OPS.Softmax:
      args = [item.arg[0], item.src[0].arg[0]]
      forward_args = item.arg[1] if len(item.arg) > 1 else ()
      axis = forward_args[0] if len(forward_args) > 0 else -1
      kernel_code, kernel_name, rows = self.renderer.softmax(dtypes.float32, args, axis=axis)
      if DEBUG >= 2: print('\n', kernel_code, '\n')

      if self.mngr.dev_name == "CUDA": kernel_name = kernel_name.encode("utf-8")
      kfunc = self.mngr.compile_kernel(kernel_code, kernel_name)
      elapsed_ms, gflops = self.run_rows_kernel(kfunc, args, rows=rows, n_flops=int(np.prod(args[0].shape)) * 3)
      if viz.enabled():
        grid, block = self.launch_dims(rows)
        viz.record_kernel(name=kernel_name, source=kernel_code, device=self.device_name(), schedule_id=id, uop_summary=_uop_summary(item), uop_repr=_uop_repr(item), op=item.op, args=args, shape=args[0].shape, axis=axis, rows=rows, grid=grid, block=block, memory_bytes=sum(uop.arg[0].nbytes for uop in item.src), elapsed_ms=elapsed_ms, gflops=gflops)
      if DEBUG >= 1:
        debug_str = f"{self.debug_prefix(id)} {color_red(kernel_name)}  {len(item.src) } {item.dtype.name}   arg {len(item.arg) if item.arg else 0}   mem {sum(uop.arg[0]._data.nbytes for uop in item.src) / (1024**3):.6f} GB"
        debug_str += f"   ({elapsed_ms:.4f} ms - {gflops:.4f} GFLOPs)   softmax"
        print(debug_str)
      return

    if item.op in REDUCE_OPS:
      reduce_map = {
        OPS.SUM: OPS.ADD,
        # TODO: all reduce patterns
      }
      # args: (output, input)
      out_tensor = item.arg[0]
      in_tensor = item.src[0].arg[0]
      out_shape = out_tensor.shape
      in_shape = in_tensor.shape

      forward_args = item.arg[1] if len(item.arg) > 1 else ()
      forward_kwargs = item.arg[2] if len(item.arg) > 2 else {}
      axis, keepdims = None, None
      if len(forward_args) == 1:   axis = forward_args[0]
      elif len(forward_args) >= 2: axis, keepdims = forward_args
      dim = axis
      if dim < 0: dim += len(in_tensor.shape)

      kernel_code, kernel_name, out_shape = self.renderer.reduce(reduce_map[item.op], dtypes.float32, [out_tensor, in_tensor], shape=in_tensor.shape, dim=dim)
      if DEBUG >= 2: print("\n", kernel_code, "\n")

      if self.mngr.dev_name == "CUDA": kernel_name = kernel_name.encode("utf-8")
      kfunc = self.mngr.compile_kernel(kernel_code, kernel_name)
      elapsed_ms, gflops = self.run_reduce_kernel(kfunc, [out_tensor, in_tensor], out_shape)
      if viz.enabled():
        grid, block = self.launch_dims(int(np.prod(out_shape)))
        viz.record_kernel(name=kernel_name, source=kernel_code, device=self.device_name(), schedule_id=id, uop_summary=_uop_summary(item), uop_repr=_uop_repr(item), op=item.op, args=[out_tensor, in_tensor], input_shape=in_shape, output_shape=out_shape, axis=axis, grid=grid, block=block, memory_bytes=in_tensor.nbytes, elapsed_ms=elapsed_ms, gflops=gflops)
      if DEBUG >= 1: print(f"{self.debug_prefix(id)} {color_red(kernel_name)} reduce ({elapsed_ms:.4f} ms - {gflops:.4f} GFLOPs)")
      return

  def run_elementwise_kernel(self, kfunc, args: list, shape: tuple, contiguous: bool):
    kargs = self.mngr.prep_kargs(*[arg.device_data if hasattr(arg, "device_data") else arg for arg in args])
    numel = int(np.prod(shape))
    if numel > self.mngr.max_grid_size[0] * self.mngr.max_block_size[0]:
      raise ValueError(f"Kernel launch failed: numel {numel} exceeds device's max grid size {self.mngr.max_grid_size[0]}. Consider implementing tiling for large tensors.")
    grid, block = self.launch_dims(numel)
    n_flops = int(np.prod(shape))
    return self.mngr.launch_kernel(kfunc, grid, block, kargs, n_flops=n_flops)

  def run_unary_kernel(self, kfunc, args: list, shape: tuple):
    kargs = self.mngr.prep_kargs(*[arg.device_data if hasattr(arg, "device_data") else arg for arg in args])
    numel = int(np.prod(shape))
    if numel > self.mngr.max_grid_size[0] * self.mngr.max_block_size[0]:
      raise ValueError(f"Kernel launch failed: numel {numel} exceeds device's max grid size {self.mngr.max_grid_size[0]}. Consider implementing tiling for large tensors.")
    grid, block = self.launch_dims(numel)
    n_flops = int(np.prod(shape))
    return self.mngr.launch_kernel(kfunc, grid, block, kargs, n_flops=n_flops)

  def run_rows_kernel(self, kfunc, args: list, rows: int, n_flops: int):
    kargs = self.mngr.prep_kargs(*[arg.device_data if hasattr(arg, "device_data") else arg for arg in args])
    grid, block = self.launch_dims(rows)
    return self.mngr.launch_kernel(kfunc, grid, block, kargs, n_flops=n_flops)

  def run_reduce_kernel(self, kfunc, args: list, out_shape: tuple):
    kargs = self.mngr.prep_kargs(*[arg.device_data if hasattr(arg, "device_data") else arg for arg in args])
    numel = int(np.prod(out_shape))
    if numel > self.mngr.max_grid_size[0] * self.mngr.max_block_size[0]:
      raise ValueError(f"Kernel launch failed: numel {numel} exceeds device's max grid size {self.mngr.max_grid_size[0]}. Consider implementing tiling for large tensors.")
    grid, block = self.launch_dims(numel)
    n_flops = int(np.prod(out_shape)) # FLOPs ≈ output_elems × reduce_dim
    return self.mngr.launch_kernel(kfunc, grid, block, kargs, n_flops=n_flops)

  def launch_dims(self, numel: int):
    if self.mngr.dev_name == "CUDA":
      block_x = min(256, self.mngr.max_block_size[0], max(1, int(numel)))
      return ((int(numel) + block_x - 1) // block_x, 1, 1), (block_x, 1, 1)
    return (int(numel), 1, 1), (self.mngr.max_block_size[0], 1, 1)
