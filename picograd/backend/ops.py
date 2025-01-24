import numpy as np
from enum import Enum, auto

from picograd.util import *


class OPS(Enum):
  # Binary
  ADD = auto()
  MUL = auto()
  DOT = auto()
  POW = auto()

  # Unary
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

  Reshape = auto()
  Flatten = auto()
  Unsqueeze = auto()

  def __str__(self): return self.name


# TODO: create @jit decorator that uses jit.compile() to execute the op
class BinaryOps:
  @staticmethod
  def add(a: np.ndarray, b: np.ndarray) -> np.ndarray: return a + b

  @staticmethod
  def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray: return a * b

  @staticmethod
  def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray: return a @ b

  @staticmethod
  def conv2d(a: np.ndarray, w: np.ndarray, b: np.array,
             in_channels: int, out_channels: int, stride: int = 1, padding: int = 0,
             debug=False) -> np.ndarray:
    # TODO: c + cbuild()
    # TODO: use C instead of in_channels
    BS, C, H, W = a.shape
    kernel_size = w.shape[1]
    H_out = ((H - kernel_size + 2*padding) // stride) + 1
    W_out = ((W - kernel_size + 2*padding) // stride) + 1
    out = np.zeros((BS, out_channels, H_out, W_out))

    for batch in range(BS):
      for out_c in range(out_channels):
        for in_c in range(in_channels):
          i_idx = 0 - padding
          for i in range(H_out):
            j_idx = 0 - padding
            for j in range(W_out):
              for k in range(kernel_size):
                for l in range(kernel_size):
                  if i_idx + k >= 0 and j_idx + l >= 0 and i_idx + k < H and j_idx + l < W:
                    out[batch][out_c][i][j] += b[out_c]
                  out[batch][out_c][i][j] += a[batch][in_c][i_idx + k][j_idx + l] * w[out_c][k][l] + b[out_c]
    return out

  @staticmethod
  def conv2d_backward(a: np.ndarray, grad_out: np.ndarray, w: np.ndarray, b: np.ndarray,
                      in_channels: int, out_channels: int, stride: int = 1, padding: int = 0):
    BS, C, H, W = a.shape
    kernel_size = w.shape[1]
    H_out = ((H - kernel_size + 2*padding) // stride) + 1
    W_out = ((W - kernel_size + 2*padding) // stride) + 1

    grad_a = np.zeros_like(a)
    grad_w = np.zeros_like(w)
    grad_b = np.zeros_like(b)

    for batch in range(BS):
      for out_c in range(out_channels):
        for in_c in range(in_channels):
          i_idx = 0 - padding
          for i in range(H_out):
            j_idx = 0 - padding
            for j in range(W_out):
              for k in range(kernel_size):
                for l in range(kernel_size):
                  if i_idx + k >= 0 and j_idx + l >= 0 and i_idx + k < H and j_idx + l < W:
                    grad_a[batch][in_c][i_idx + k][j_idx + l] += grad_out[batch][out_c][i][j] * w[out_c][k][l]
                    grad_w[out_c][k][l] += grad_out[batch][out_c][i][j] * a[batch][in_c][i_idx + k][j_idx + l]
                    grad_b[out_c] += grad_out[batch][out_c][i][j]
    return grad_a, grad_w, grad_b
class UnaryOps:
  @staticmethod
  def relu(a: np.ndarray) -> np.ndarray: return np.maximum(a, np.zeros_like(a))
  
  @staticmethod
  def sigmoid(a: np.ndarray) -> np.ndarray: pass
  
  @staticmethod
  def tanh(a: np.ndarray) -> np.ndarray: pass
  
  @staticmethod
  def abs(a: np.ndarray) -> np.ndarray: pass
  
  @staticmethod
  def neg(a: np.ndarray) -> np.ndarray: pass
  
  @staticmethod
  def sqrt(a: np.ndarray) -> np.ndarray: pass
  
  @staticmethod
  def exp(a: np.ndarray) -> np.ndarray: pass
  
  @staticmethod
  def log(a: np.ndarray) -> np.ndarray: pass
  
  @staticmethod
  def normalize(a: np.ndarray) -> np.ndarray: pass
  
  @staticmethod
  def softmax(a: np.ndarray) -> np.ndarray: pass
  
  @staticmethod
  def batchnorm(a: np.ndarray) -> np.ndarray: pass


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

  
  def ufix(self, x) -> np.ndarray: return self.const_like(x) if not isinstance(x, MathTrait) else x
  def _binop(self, op, x, reverse) -> np.ndarray: return self.ufix(x).alu(op, self) if reverse else self.alu(op, self.ufix(x))
  def logical_not(self) -> np.ndarray: return self.ne(True)
  def neg(self) -> np.ndarray:
    dtype: Optional[DType] = getattr(self, 'dtype', None)
    assert dtype is not None, "MathTraits __neg__ requires a dtype"
    return self.logical_not() if dtype.scalar() == dtypes.bool else self*(-1)
  def add(self, x, reverse=False) -> np.ndarray: return self._binop(Ops.ADD, x, reverse)
  def mul(self, x, reverse=False) -> np.ndarray: return self._binop(Ops.MUL, x, reverse)
  def bitwise_and(self, x, reverse=False) -> np.ndarray: return self._binop(Ops.AND, x, reverse)
  def bitwise_or(self, x, reverse=False) -> np.ndarray: return self._binop(Ops.OR, x, reverse)
  def xor(self, x, reverse=False) -> np.ndarray: return self._binop(Ops.XOR, x, reverse)
  def idiv(self, x, reverse=False) -> np.ndarray: return self._binop(Ops.IDIV, x, reverse)
  def sub(self, x, reverse=False) -> np.ndarray: return self.ufix(x).alu(Ops.ADD, -self) if reverse else self.alu(Ops.ADD, self.ufix(-x))
  def div(self, x, reverse=False) -> np.ndarray: return (self.ufix(x)*self.alu(Ops.RECIP)) if reverse else (self*self.ufix(x).alu(Ops.RECIP))

  def __neg__(self) -> np.ndarray: return self.neg()

  def __add__(self, x) -> np.ndarray: return self.add(x)
  def __sub__(self, x) -> np.ndarray: return self.sub(x)
  def __mul__(self, x) -> np.ndarray: return self.mul(x)
  def __truediv__(self, x) -> np.ndarray: return self.div(x)
  def __floordiv__(self, x) -> np.ndarray: return self.idiv(x)
  def __and__(self, x) -> np.ndarray: return self.bitwise_and(x)
  def __or__(self, x) -> np.ndarray: return self.bitwise_or(x)
  def __xor__(self, x) -> np.ndarray: return self.xor(x)
"""

