from enum import Enum, auto

from picograd.util import *
from .function import *


class OPS(Enum):
  # Binary
  ADD = auto()
  MUL = auto()
  POW = auto()

  Flatten = auto()
  Unsqueeze = auto()

  ReLU = auto()
  Tanh = auto()
  Softmax = auto()
  Sigmoid = auto()

  MSELoss = auto()
  MAELoss = auto()
  CrossEntropyLoss = auto()
  BCELoss = auto()

  Conv2D = auto()
  MaxPool2D = auto()
  AvgPool2D = auto()

  def __str__(self):
    return self.name


# TODO: create @jit decorator that uses jit.compile() to execute the op
class BinaryOps:
  @staticmethod
  def add(a, b): return a + b

  @staticmethod
  def mul(a, b): return a * b

# TODO:
"""
TINYGRAD:
class GroupOp:
  Unary = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIP, Ops.NEG}
  Binary = {Ops.ADD, Ops.MUL, Ops.IDIV, Ops.MAX, Ops.MOD, Ops.CMPLT, Ops.CMPNE, Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.THREEFRY,
            Ops.SUB, Ops.FDIV}
  Ternary = {Ops.WHERE, Ops.MULACC}
  ALU = set.union(Unary, Binary, Ternary)

  Irreducible = {Ops.CONST, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.RANGE}

  # meta ops
  Meta = {Ops.COPY, Ops.EMPTY, Ops.BUFFER_VIEW}
  Buffer = {Ops.LOAD, Ops.PRELOAD, Ops.STORE, Ops.VALID}

  # BinaryOps that can be flipped
  Commutative = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.XOR, Ops.AND, Ops.OR}

  # do not preserve f(0) = 0
  UnsafePad = {Ops.RECIP, Ops.LOG2, Ops.EXP2, Ops.IDIV}

  
  def ufix(self, x): return self.const_like(x) if not isinstance(x, MathTrait) else x
  def _binop(self, op, x, reverse): return self.ufix(x).alu(op, self) if reverse else self.alu(op, self.ufix(x))
  def logical_not(self): return self.ne(True)
  def neg(self):
    dtype: Optional[DType] = getattr(self, 'dtype', None)
    assert dtype is not None, "MathTraits __neg__ requires a dtype"
    return self.logical_not() if dtype.scalar() == dtypes.bool else self*(-1)
  def add(self, x, reverse=False): return self._binop(Ops.ADD, x, reverse)
  def mul(self, x, reverse=False): return self._binop(Ops.MUL, x, reverse)
  def bitwise_and(self, x, reverse=False): return self._binop(Ops.AND, x, reverse)
  def bitwise_or(self, x, reverse=False): return self._binop(Ops.OR, x, reverse)
  def xor(self, x, reverse=False): return self._binop(Ops.XOR, x, reverse)
  def idiv(self, x, reverse=False): return self._binop(Ops.IDIV, x, reverse)
  def sub(self, x, reverse=False): return self.ufix(x).alu(Ops.ADD, -self) if reverse else self.alu(Ops.ADD, self.ufix(-x))
  def div(self, x, reverse=False): return (self.ufix(x)*self.alu(Ops.RECIP)) if reverse else (self*self.ufix(x).alu(Ops.RECIP))

  def __neg__(self): return self.neg()

  def __add__(self, x): return self.add(x)
  def __sub__(self, x): return self.sub(x)
  def __mul__(self, x): return self.mul(x)
  def __truediv__(self, x): return self.div(x)
  def __floordiv__(self, x): return self.idiv(x)
  def __and__(self, x): return self.bitwise_and(x)
  def __or__(self, x): return self.bitwise_or(x)
  def __xor__(self, x): return self.xor(x)
"""

