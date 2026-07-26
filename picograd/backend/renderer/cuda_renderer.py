import numpy as np

from .cstyle import CStyleRenderer
from picograd.backend.uop import UOp
from picograd.backend.function import OPS
from picograd.backend.dtypes import dtypes

class CUDARenderer(CStyleRenderer):
  device = "CUDA"
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  shared_max = 49152

  kernel_typedef = "extern \"C\" __global__ void"
  elementwise_kernel_prefix = "E"
  reduce_kernel_prefix = "R"

  # TODO: support more dims
  gidx = "gidx0"
  gidx0 = "gidx0"  
  gidx1 = "gidx1"

  block_x = "blockIdx.x"
  block_y = "blockIdx.y"

  block_dim_x = "blockDim.x"
  block_dim_y = "blockDim.y"

  tid_x = "threadIdx.x"
  tid_y = "threadIdx.y"
  # TODO: "blockIdx.x * blockDim.x + threadIdx.x"

  def __init__(self, arch:str):
    self.arch = arch
    # self.tensor_cores = tc.cuda_sm89 if int(arch[3:]) >= 89 else tc.cuda_sm80 if int(arch[3:]) >= 80 else tc.cuda_sm75 if int(arch[3:]) >= 75 else []

  def __reduce__(self):
    return self.__class__, (self.arch,)

  def get_args(args): return [f"data{i}" for i in range(len(args))]

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

    # build kernel
    func_expr = []

    # logical indices
    func_expr.append([dtypes.int32.name, self.gidx, self.assign, f"{self.block_x} * {self.block_dim_x} + {self.tid_x}"])

    # bounds checking
    numel = int(np.prod(shape))
    func_expr.append(["if", f"({self.gidx} >= {numel})", "return"])

    # func_expr.append([dtypes.int32.name, self.gidx, self.assign, self.block_x]) # int gidx0 = blockIdx.x
    if contiguous:
      func_expr.append([dtypes.int32.name, self.gidx, self.assign,f"{self.block_x}"]) # flat index
      for k, val in enumerate(vals):
        func_expr.append([dtype.name, val, self.assign, f"*({args[k+1]} + {self.gidx})"])
      func_expr.append([f"*({args[0]} + {self.gidx})", self.assign, f"({vals[0]}{alu}{vals[1]})"])
    else:
      # per-tensor offsets (reconstruct multi-index from gidx)
      tmp = "tmp"
      func_expr.append([dtypes.int32.name, tmp, self.assign, self.gidx])
      for d in reversed(range(len(shape))):
        size = shape[d]
        func_expr.append([dtypes.int32.name, f"i{d}", self.assign, f"{tmp} % {size}"])
        func_expr.append([tmp, self.assign, f"{tmp} / {size}"])

      # compute per-tensor offsets using strides
      off0 = " + ".join([f"i{d}*{s0[d]}" for d in range(len(shape))])
      off1 = " + ".join([f"i{d}*{s1[d]}" for d in range(len(shape))])
      off2 = " + ".join([f"i{d}*{s2[d]}" for d in range(len(shape))])
      func_expr.append([f"{args[0]}[{off0}]", self.assign, f"{args[1]}[{off1}]{alu}{args[2]}[{off2}]"])

    kernel = [
      self.kernel_typedef,
      kernel_name,
      self.parenthesis(", ".join([f"{dtype.name} *{arg}" for arg in args])),
      self.curly_braces('\t' + (self.end_expr+'\t').join([' '.join(expr) for expr in func_expr]) + self.semicolon)
    ]
    prg = ' '.join(kernel)
    return prg, kernel_name, contiguous

  def relu(self, dtype, arg):
    assert len(arg) == 2, f"Expected 2 arguments for relu (output, input), got {len(arg)} instead"
    out, inp = arg
    assert out.shape == inp.shape, "ReLU output and input shapes must match"
    numel = int(np.prod(out.shape))
    kernel_name = f"U_ReLU_{'_'.join(map(str, out.shape))}"
    args = ["data0", "data1"]
    func_expr = [
      [dtypes.int32.name, self.gidx, self.assign, f"{self.block_x} * {self.block_dim_x} + {self.tid_x}"],
      ["if", f"({self.gidx} >= {numel})", "return"],
      [dtype.name, "v", self.assign, f"{args[1]}[{self.gidx}]"],
      [f"{args[0]}[{self.gidx}]", self.assign, "v > 0.0f ? v : 0.0f"],
    ]
    kernel = [
      self.kernel_typedef,
      kernel_name,
      self.parenthesis(", ".join([f"{dtype.name} *{arg}" for arg in args])),
      self.curly_braces('\t' + (self.end_expr + '\t').join([' '.join(expr) for expr in func_expr]) + self.semicolon)
    ]
    return ' '.join(kernel), kernel_name

  def softmax(self, dtype, arg, axis=-1):
    assert len(arg) == 2, f"Expected 2 arguments for softmax (output, input), got {len(arg)} instead"
    out, inp = arg
    assert out.shape == inp.shape, "Softmax output and input shapes must match"
    assert len(inp.shape) == 2, "Softmax renderer currently supports 2D tensors"
    if axis is None: axis = -1
    if axis < 0: axis += len(inp.shape)
    assert axis == 1, "Softmax renderer currently supports last-axis softmax only"
    rows, cols = inp.shape
    kernel_name = f"U_Softmax_{rows}_{cols}_d{axis}"
    body = f"""
\tint row = {self.block_x} * {self.block_dim_x} + {self.tid_x};
\tif (row >= {rows}) return;
\tint base = row * {cols};
\t{dtype.name} maxv = data1[base];
\tfor (int i = 1; i < {cols}; i++) {{
\t\t{dtype.name} v = data1[base + i];
\t\tmaxv = v > maxv ? v : maxv;
\t}}
\t{dtype.name} sum = 0.0f;
\tfor (int i = 0; i < {cols}; i++) {{
\t\t{dtype.name} e = expf(data1[base + i] - maxv);
\t\tdata0[base + i] = e;
\t\tsum += e;
\t}}
\tfor (int i = 0; i < {cols}; i++) data0[base + i] = data0[base + i] / sum;
"""
    kernel = [self.kernel_typedef, kernel_name, self.parenthesis(f"{dtype.name} *data0, {dtype.name} *data1"), self.curly_braces(body)]
    return ' '.join(kernel), kernel_name, rows

  # TODO: this is naive
  # TODO: keepdims
  # TODO: reduce (sum, max, min, std, argmax, argmin)
  # e.g. reduce(ADD, axis)
  def reduce(self, op, dtype, arg, shape: tuple[int, ...], dim: int):
    """
    arg: [out_tensor, in_tensor]
    """
    assert len(arg) == 2, "reduce expects (output, input)"
    out, inp = arg
    alu = self.op_to_alu(op)
    in_shape = shape
    in_strides = inp.strides
    out_strides = out.strides

    reduce_size = in_shape[dim]
    out_shape = in_shape[:dim] + in_shape[dim+1:]

    kernel_name = f"{self.reduce_kernel_prefix}_{op.name}_{'_'.join(map(str, in_shape))}_d{dim}"
    args = ["data0", "data1"]  # out, in

    # identity values
    identity = {
      OPS.ADD: "0.0f",
      OPS.MUL: "1.0f",
      OPS.MAX: "-INFINITY",
      OPS.MIN: "INFINITY",
    }[op]

    # compute output linear index
    # gidx0 in [0, prod(out_shape))
    func = []
    func.append([dtypes.int32.name, "oidx", self.assign, self.block_x])
    func.append([dtype.name, "acc", self.assign, identity])

    # reconstruct multi-dim output index
    func.append([dtypes.int32.name, "tmp", self.assign, "oidx"])
    func.append([dtypes.int32.name, "in_base", self.assign, "0"])

    for d in reversed(range(len(in_shape))):
      if d == dim: continue
      out_d = d if d < dim else d - 1
      size = out_shape[out_d]
      stride = in_strides[d]
      func.append([dtypes.int32.name, f"i{d}", self.assign, f"tmp % {size}"])
      func.append(["tmp", self.assign, f"tmp / {size}"])
      func.append(["in_base", self.assign, f"in_base + i{d} * {stride}"])

    # reduction loop
    func.append([self.for_loop(var="r", start="0", end=str(reduce_size), body=f"\t\tacc = acc {alu} data1[in_base + r * {in_strides[dim]}];")])

    # store result
    func.append(["data0[oidx]", self.assign, "acc"])

    kernel = [
      self.kernel_typedef,
      kernel_name,
      self.parenthesis(f"{dtype.name} *data0, {dtype.name} *data1"),
      self.curly_braces(
        "\t" + (self.end_expr + "\t").join(
          [" ".join(x) for x in func if isinstance(x, list)]
        ) + self.semicolon
      )
    ]
    prg = " ".join(kernel)
    return prg, kernel_name, out_shape

  # TODO: check out [ https://mesozoic-egg.github.io/tinygrad-notes/20241112_pm.html ]
  # TODO: use class UPat like tinygrad (cleaner than if statements)
  def render_code(self, uop: UOp):
    patterns = [
      (OPS.STORE, lambda uop: "="),
      (OPS.CONST, lambda uop: f" {uop.arg} "),
      (OPS.ADD, lambda uop: f" + "),
    ]
    code = []
    for _uop in uop: # Suppose you already did a DFS/BFS so the tree is flattened
      for pattern in patterns:
        if _uop.op == pattern[0]:
          _code = pattern[1](_uop)
          code.append(_code) 
    return code
