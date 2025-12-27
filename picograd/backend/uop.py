from __future__ import annotations
import os
from typing import Any

from .function import OPS
from .dtypes import dtypes


DEBUG = int(os.getenv("DEBUG", 0))

class UOp:
  """ Intermediate representation in the compiler """
  def __init__(self, op: OPS, dtype: dtypes, src: tuple[UOp, ...] = tuple(), arg: Any = None, tag: Any = None):
    self.op = op
    self.dtype = dtype
    self.src = src
    self.arg = arg
    self.tag = tag
  # TODO: pretty_print UOp.src
  def __repr__(self): return f"{type(self).__name__}({self.op}, {self.dtype}, arg={self.argstr()}{self.tagstr()}, src=({self.src}))"
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