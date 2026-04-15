import numpy as np

from .cstyle import CStyleRenderer
from picograd.backend.uop import UOp
from picograd.backend.function import OPS
from picograd.backend.dtypes import dtypes

class MetalRenderer(CStyleRenderer):
  device = "METAL"
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  shared_max = 32768

  kernel_typedef = "kernel void"
  elementwise_kernel_prefix = "E"
  reduce_kernel_prefix = "R"

  const = "constant"
  device = "device"
  buffer = "buffer"

  # TODO: uint id [[ thread_position_in_grid ]], uint num_threads [[ threads_per_grid ]]
  tid = "uint id [[ thread_position_in_grid ]]"

  def __init__(self):
    pass

  def __reduce__(self):
    return self.__class__, (self.arch,)
  
  def get_args(args): return [f"data{i}" for i in range(len(args))]

  def render_code(self, uop: UOp):
    patterns = [
      (OPS.STORE, lambda uop: "="),
      (OPS.CONST, lambda uop: f" {uop.arg}"),
      (OPS.ADD, lambda uop: " + "),
    ]
    code = []
    for _uop in uop:
      for pattern, render in patterns:
        if _uop.op == pattern:
          code.append(render(_uop))
          break # TODO: is this correct?
      else:
        raise NotImplementedError(f"Unsupported operation: {_uop.op}")
    return code

  def elementwise(self, op, dtype, arg):
    assert len(arg) == 3, f"Expected 3 arguments for c = a alu b, got {len(arg)} instead"
    assert all(a.shape == arg[0].shape for a in arg), "All input shapes must match for elementwise ops"

    contiguous = all(a.is_contiguous for a in arg)
    alu = self.op_to_alu(op)
    shape = arg[0].shape
    strides = [a.strides for a in arg]
    s0 = strides[0]
    s1 = strides[1]
    s2 = strides[2]

    size_t = '_'.join([str(s) for s in shape])
    kernel_name = f"{self.elementwise_kernel_prefix}_{op.name}_{size_t}"
    args = [f"data{i}" for i in range(len(arg))]
    vals = [f"val{i}" for i in range(len(arg)-1)]

    func_expr = []

    # bounds check
    numel = int(np.prod(shape))
    func_expr.append(["if", f"(id >= {numel})", "return"])

    if contiguous:
      func_expr.append([f"{args[0]}[id]", self.assign, f"{args[1]}[id]", alu, f"{args[2]}[id]"])
    else:
      tmp = "tmp"
      func_expr.append([dtypes.int32.name, tmp, self.assign, "id"])

      # reconstruct multi-dim indices
      for d in reversed(range(len(shape))):
        size = shape[d]
        func_expr.append([dtypes.int32.name, f"i{d}", self.assign, f"{tmp} % {size}"])
        func_expr.append([tmp, self.assign, f"{tmp} / {size}"])

      # compute offsets using strides
      off0 = " + ".join([f"i{d}*{s0[d]}" for d in range(len(shape))])
      off1 = " + ".join([f"i{d}*{s1[d]}" for d in range(len(shape))])
      off2 = " + ".join([f"i{d}*{s2[d]}" for d in range(len(shape))])

      func_expr.append([f"{args[0]}[{off0}]", self.assign, f"{args[1]}[{off1}]", alu, f"{args[2]}[{off2}]"])

    arg_list = []
    arg_list += [f"{self.device} {dtype.name} *{args[0]} [[ {self.buffer}(0) ]]"]
    arg_list += [f"{self.const} {dtype.name} *{args[i]} [[ {self.buffer}({i}) ]]" for i in range(1, len(args))]
    arg_list += [self.tid]

    kernel = [
      self.kernel_typedef,
      kernel_name,
      self.parenthesis(", ".join(arg_list)),
      self.curly_braces(
        '\t' + (self.end_expr + '\t').join([' '.join(expr) for expr in func_expr]) + self.semicolon
        if func_expr else ''
      )
    ]
    prg = ' '.join(kernel)
    return prg, kernel_name, contiguous

  # TODO: cleanup using tokens only, not string manipulation
  # TODO: this might be naive reduction (not using shared memory, etc)
  # TODO: keepdims
  def reduce(self, op, dtype, arg, shape: tuple[int, ...], dim: int):
    assert len(arg) == 2, f"Expected 2 arguments for reduction (output, input), got {len(arg)} instead"
    out, inp = arg
    alu = self.op_to_alu(op)
    in_shape = shape
    in_strides = inp.strides

    reduce_size = in_shape[dim]
    out_shape = in_shape[:dim] + in_shape[dim+1:]

    kernel_name = f"{self.reduce_kernel_prefix}_{op.name}_{'_'.join(map(str, in_shape))}_d{dim}"

    identity = {
      OPS.ADD: "0.0f",
      OPS.MUL: "1.0f",
      OPS.MAX: "-INFINITY",
      OPS.MIN: "INFINITY",
    }[op]

    func = []

    # oidx = global thread id
    func.append([dtypes.int32.name, "oidx", self.assign, "id"])

    # bounds check
    numel = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
    func.append(["if", f"(oidx >= {numel})", "return"])

    # accumulator
    func.append([dtype.name, "acc", self.assign, identity])

    # reconstruct base offset (excluding reduced dim)
    func.append([dtypes.int32.name, "tmp", self.assign, "oidx"])
    func.append([dtypes.int32.name, "in_base", self.assign, "0"])

    # iterate over input dims (skip reduction dim)
    for d in reversed(range(len(in_shape))):
      if d == dim:
        continue

      size = in_shape[d]
      stride = in_strides[d]

      func.append([dtypes.int32.name, f"i{d}", self.assign, f"tmp % {size}"])
      func.append(["tmp", self.assign, f"tmp / {size}"])
      func.append(["in_base", self.assign, f"in_base + i{d} * {stride}"])

    # reduction loop (walk along reduced dimension)
    func.append(
      self.for_loop(
        "r",
        "0",
        str(reduce_size),
        f"acc = acc {alu} data1[in_base + r * {in_strides[dim]}]{self.semicolon}"
      )
    )

    # write output
    func.append(["data0[oidx]", self.assign, "acc"])

    kernel = [
      self.kernel_typedef,
      kernel_name,
      self.parenthesis(
        f"{self.device} {dtype.name} *data0 [[ buffer(0) ]], "
        f"{self.const} {dtype.name} *data1 [[ buffer(1) ]], "
        f"{self.tid}"
      ),
      self.curly_braces(
        "\t" + (self.end_expr + "\t").join(
          [" ".join(x) if isinstance(x, list) else x for x in func]
        ) + self.semicolon
      )
    ]
    prg = " ".join(kernel)
    return prg, kernel_name, out_shape