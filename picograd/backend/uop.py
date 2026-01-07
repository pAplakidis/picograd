from __future__ import annotations
import os
from typing import Any, Callable

from .function import OPS
from .dtypes import dtypes


DEBUG = int(os.getenv("DEBUG", 0))

# NOTE: UOp is a singleton - this checks if two UOps are identical (same op, dtype, src, arg) and reuses them
# class UOpMetaClass(type):
#   ucache:dict[tuple, weakref.ReferenceType[UOp]] = {}
#   def __call__(cls, op:Ops, dtype:DType=dtypes.void, src:tuple[UOp,...]=tuple(), arg:Any=None, _buffer:Buffer|None=None):
#     if (wret:=UOpMetaClass.ucache.get(key:=(op, dtype, src, arg), None)) is not None and (ret:=wret()) is not None: return ret
#     UOpMetaClass.ucache[key] = ref = weakref.ref(created:=super().__call__(*key))

def pretty_print(x:Any, rep:Callable, srcfn=lambda x: x.src, cache=None, d=0)->str:
  def dfs(x:Any, cache:dict):
    for s in srcfn(x) or []:
      cache.setdefault(s, [len(cache), 0, False])[1] += 1
      if cache[s][1] == 1: dfs(s, cache)
  if cache is None: dfs(x, cache:={})
  if (cx:=cache.setdefault(x, [0,0,False]))[2]: return f"{' '*d} x{cx[0]}"
  cx[2], srcs = True, ('None' if srcfn(x) is None else ''.join(f'\n{pretty_print(s, rep, srcfn, cache, d+2)},' for s in srcfn(x)))
  return f"{' '*d}{f'x{cx[0]}:=' * (cx[1]>1)}{rep(x)}" % srcs

class UOp:
  """ Intermediate representation in the compiler """
  def __init__(self, op: OPS, dtype: dtypes, src: tuple[UOp, ...] = tuple(), arg: Any = None, tag: Any = None):
    self.op = op
    self.dtype = dtype
    self.src = src
    self.arg = arg
    self.tag = tag
  def __repr__(self): return pretty_print(self, lambda x: f"{type(self).__name__}({x.op}, {x.dtype}, arg={x.argstr()}{x.tagstr()}, src=(%s))")
  def tagstr(self): return f", tag={self.tag}" if self.tag is not None else ""
  def argstr(self):
    if self.arg is None:
      return "None"

    if DEBUG >= 1:
      return repr(self.arg)

    # non-debug, hide object identities
    if isinstance(self.arg, (tuple, list)):
      return "[" + ", ".join(
        f"<{type(a).__module__}.{type(a).__qualname__} at {hex(id(a))}>"
        for a in self.arg
      ) + "]"

    # scalar / non-iterable arg
    return f"<{type(self.arg).__name__}>"