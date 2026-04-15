from typing import Literal, Callable

from picograd.backend.dtypes import DType
from picograd.backend.function import OPS

class CStyleRenderer():
  kernel_typedef: str = "void"
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = "const int"
  barrier: str = ""
  code_for_workitem: dict[Literal["g", "l", "i"], Callable] = {}
  extra_args: list[str] = []
  float4: str|None = None
  float4_style: tuple[str, str] = ('(', ')')
  gep_arr_threshold: int = 4
  type_map: dict[DType, str] = {}
  infinity: str = "INFINITY"
  nan: str = "NAN"

  assign = "="
  semicolon = ";"
  end_expr = f"{semicolon}\n"

  elementwise_ops = [OPS.ADD, OPS.MUL, OPS.POW]
  binary_ops = [OPS.DOT, OPS.Conv2D]  # TODO: shape trick => elementwise

  @staticmethod
  def parenthesis(expr: str) -> str: return f"({expr})"

  @staticmethod
  def curly_braces(expr: str) -> str: return f"{{\n{expr}\n}}"

  @staticmethod
  def square_brackets(expr: str) -> str: return f"[{expr}]"

  @staticmethod
  def for_loop(var: str, start: str, end: str, body: str) -> str: return ' '.join([f"for (int {var} = {start}; {var} < {end}; {var}++)", CStyleRenderer.curly_braces(body)])

  def op_to_alu(self, op: OPS) -> str:
    if op == OPS.ADD: return " + "
    if op == OPS.MUL: return " * "

    raise NotImplementedError(f"Unsupported op: {op}")

def elementwise(op, dtype, arg):
  prg = """#include <stddef.h>
  float E_4_4(float* data0, float* data1, float* data2, size_t N){
    for (size_t gidx0 = 0; gidx0 < N; gidx0++) {
      float val0 = data1[gidx0];
      float val1 = data2[gidx0];
      data0[gidx0] = val0 + val1;
    }
  }"""
  return prg
