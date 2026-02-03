from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final, ClassVar, Callable, Literal

FmtStr = Literal['?', 'b', 'B', 'h', 'H', 'i', 'I', 'q', 'Q', 'e', 'f', 'd']

class AddrSpace(Enum):
  def __repr__(self): return str(self)
  GLOBAL = auto(); LOCAL = auto(); REG = auto()  # noqa: E702

class DTypeMetaClass(type):
  dcache: dict[tuple, DType] = {}
  def __call__(cls, *args, **kwargs):
    if (ret:=DTypeMetaClass.dcache.get(args, None)) is not None: return ret
    DTypeMetaClass.dcache[args] = ret = super().__call__(*args)
    return ret

@dataclass(frozen=True, eq=True)
class DType:
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  fmt: FmtStr|None
  count: int
  _scalar: DType|None
  @staticmethod
  def new(priority:int, itemsize:int, name:str, fmt:FmtStr|None): return DType(priority, itemsize, name, fmt, 1, None)
  def __repr__(self): return f"dtypes.{INVERSE_DTYPES_DICT[self.scalar().name]}"+(f".vec({self.count})" if self.count > 1 else "")
  def scalar(self) -> DType: return self._scalar if self._scalar is not None else self

# TODO: connect with Tensor.dtype, numpy.dtypes, etc
class dtypes:
  void: Final[DType]    = DType.new(-1, 0, "void", None)
  index: Final[DType]   = DType.new(-1,100, "index", None)
  bool: Final[DType]    = DType.new(0, 1, "bool", '?')
  int8: Final[DType]    = DType.new(1, 1, "signed char", 'b')
  uint8: Final[DType]   = DType.new(2, 1, "unsigned char", 'B')
  int16: Final[DType]   = DType.new(3, 2, "short", 'h')
  uint16: Final[DType]  = DType.new(4, 2, "unsigned short", 'H')
  int32: Final[DType]   = DType.new(5, 4, "int", 'i')
  uint32: Final[DType]  = DType.new(6, 4, "unsigned int", 'I')
  int64: Final[DType]   = DType.new(7, 8, "long", 'q')
  uint64: Final[DType]  = DType.new(8, 8, "unsigned long", 'Q')
  fp8e4m3: Final[DType] = DType.new(9, 1, "float8_e4m3", None)
  fp8e5m2: Final[DType] = DType.new(10, 1, "float8_e5m2", None)
  float16: Final[DType] = DType.new(11, 2, "half", 'e')
  # bfloat16 has higher priority than float16, so least_upper_dtype(dtypes.int64, dtypes.uint64) = dtypes.float16
  bfloat16: Final[DType] = DType.new(12, 2, "__bf16", None)
  float32: Final[DType] = DType.new(13, 4, "float", 'f')
  float64: Final[DType] = DType.new(14, 8, "double", 'd')

DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if isinstance(v, DType) and not k.startswith(("default", "void", "index"))}
INVERSE_DTYPES_DICT = {**{v.name:k for k,v in DTYPES_DICT.items()}, "void": "void", "index":"index"}
