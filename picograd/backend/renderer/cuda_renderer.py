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
  elementwise_kernel_name = "E"

  # TODO: support more dims
  gidx = "gidx0"  
  block_x = "blockIdx.x"
  block_dim = "blockDim.x"
  tid_x = "threadIdx.x"
  # TODO: "blockIdx.x * blockDim.x + threadIdx.x"

  def __init__(self, arch:str):
    self.arch = arch
    # self.tensor_cores = tc.cuda_sm89 if int(arch[3:]) >= 89 else tc.cuda_sm80 if int(arch[3:]) >= 80 else tc.cuda_sm75 if int(arch[3:]) >= 75 else []

  def __reduce__(self):
    return self.__class__, (self.arch,)

  def get_args(args): return [f"data{i}" for i in range(len(args))]

  def elementwise(self, op, dtype, arg, shape: tuple[int,...]):
    alu = self.op_to_alu(op)
    size_t = '_'.join([str(s) for s in shape])
    kernel_name = f"{self.elementwise_kernel_name}_{op.name}_{size_t}"
    args = [f"data{i}" for i in range(len(arg))]
    vals = [f"val{i}" for i in range(len(arg)-1)]

    func_expr = []
    # offset
    func_expr.append([dtypes.int32.name, self.gidx, self.assign, self.block_x])                       # int gidx0 = blockIdx.x
    # load vals
    for val in vals:
      func_expr.append([dtype.name, val, self.assign, f"*({args[vals.index(val)+1]} + {self.gidx})"]) # dtype vali = *(datai + gidx0)
    # alu and store
    func_expr.append([f"*({args[0]} + {self.gidx})", self.assign, f"({vals[0]}{alu}{vals[1]})"])    # *(data0 + gidx0) = (val0 alu val1)

    kernel = [
      self.kernel_typedef,
      kernel_name,
      self.parenthesis(", ".join([f"{dtype.name} *{arg}" for arg in args])),
      self.curly_braces('\t' + (self.end_expr+'\t').join([' '.join(expr) for expr in func_expr]) + self.semicolon)
    ]
    prg = ' '.join(kernel)
    return prg, kernel_name

  # TODO: reduce (sum, max, min, std, argmax, argmin)
  def reduce(self, op, dtype, arg, shape: tuple[int,...], dim: int):
    raise NotImplementedError("CUDA reduce not implemented yet")

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
