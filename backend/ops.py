import numpy as np
from enum import Enum, auto

from util import *

class OPS(Enum):
  Linear = auto()
  Conv2D = auto()
  ReLU = auto()
  Tanh = auto()
  Softmax = auto()
  Sigmoid = auto()
  MSELoss = auto()
  MAELoss = auto()
  CrossEntropyLoss = auto()
  BCELoss = auto()
  MaxPool2D = auto()
  AvgPool2D = auto()
  Flatten = auto()
  Unsqueeze = auto()

  def __str__(self):
    return self.name


# TODO: ADD, MUL
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
"""

