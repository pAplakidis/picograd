import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Tuple

from picograd.util import *

def reduce_grad(grad: np.ndarray, shape: tuple) -> np.ndarray:
  """Reduce broadcasted gradient `grad` back to `shape`."""

  # Step 1: collapse extra leading dims
  while grad.ndim > len(shape): grad = grad.sum(axis=0)

  # Step 2: sum over axes where original shape had 1 (broadcasted dims)
  for axis, dim in enumerate(shape):
    if dim == 1: grad = grad.sum(axis=axis, keepdims=True)

  return grad


# TODO: create @jit decorator that uses jit.compile() to execute the op (LLVM)
class BinaryOps:
  @staticmethod
  def add(a: "Tensor", b: "Tensor") -> np.ndarray: return a.data + b.data

  @staticmethod
  def add_back(a: "Tensor", b: "Tensor", grad_out: np.ndarray):
    if a.requires_grad: a.grad += reduce_grad(grad_out, a.data.shape)
    if b.requires_grad: b.grad += reduce_grad(grad_out, b.data.shape)

  @staticmethod
  def mul(a: "Tensor", b: "Tensor") -> np.ndarray: return a.data * b.data

  @staticmethod
  def mul_back(a: "Tensor", b: "Tensor", grad_out: np.ndarray):
    if a.requires_grad: a.grad += reduce_grad(b.data * grad_out, a.data.shape)
    if b.requires_grad: b.grad += reduce_grad(a.data * grad_out, b.data.shape)

  @staticmethod
  def dot(a: "Tensor", b: "Tensor") -> np.ndarray: return a.data @ b.data

  @staticmethod
  def dot_back(a: "Tensor", b: "Tensor", grad_out: np.ndarray):
    if a.requires_grad: a.grad += grad_out @ b.data.T
    if b.requires_grad: b.grad += a.data.T @ grad_out

  @staticmethod
  def pow(a: "Tensor", b: "Tensor") -> np.ndarray: return a.data ** b.data

  @staticmethod
  def pow_back(a: "Tensor", b: "Tensor", grad_out: np.ndarray):
    if a.requires_grad: a.grad += b.data * (a.data ** (b.data - 1)) * grad_out

  @staticmethod
  def conv2d(A: "Tensor", Weight: "Tensor", Bias: "Tensor",
             in_channels: int, out_channels: int, stride: int = 1, padding: int = 0,
             debug=False) -> np.ndarray:
    a = A.data
    w = Weight.data
    b = Bias.data

    assert a.shape[1] == in_channels, "Input channels do not match"
    assert len(a.shape) == 4, "Input must be 4D (B, C, H, W)"
    assert len(w.shape) == 4, "Kernel must be 4D (C_out, C_in, H, W)"
    assert w.shape[2] == w.shape[3], "Kernel must be square"
    assert w.shape[1] % 2 == 1, "Kernel dimensions must be odd"
    assert a.shape[2] >= w.shape[1] and a.shape[3] >= w.shape[2], "Input must be larger than or equal to kernel dimensions"

    def im2col(a, kernel_size, stride):
      BS, C, H, W = a.shape
      H_out = (H - kernel_size)//stride + 1
      W_out = (W - kernel_size)//stride + 1

      shape = (BS, C, H_out, W_out, kernel_size, kernel_size)
      strides = (
        a.strides[0],
        a.strides[1],
        stride * a.strides[2],
        stride * a.strides[3],
        a.strides[2],
        a.strides[3],
      )
      return as_strided(a, shape=shape, strides=strides)

    BS, C_in, H, W = a.shape
    C_out, _, kernel_size, _ = w.shape

    # init output
    H_out = ((H - kernel_size + 2*padding) // stride) + 1
    W_out = ((W - kernel_size + 2*padding) // stride) + 1
    out = np.zeros((BS, out_channels, H_out, W_out))

    # add padding
    if padding > 0:
      a_padded = np.pad(a, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
      a_padded = a

    cols = im2col(a_padded, kernel_size, stride)    # (BS, in_c, H_out, W_out, k, k)
    cols = cols.reshape(BS, in_channels, H_out, W_out, -1).transpose(0,2,3,1,4).reshape(BS, H_out*W_out, -1)  # (BS, H_out * W_out, in_c * k * k)
    w_flat = w.reshape(out_channels, -1)            # (out_c, in_c * k * k)
    out_flat = cols @ w_flat.T                      # (BS, H_out*W_out, out_c)
    out = out_flat.transpose(0,2,1).reshape(BS, out_channels, H_out, W_out)
    out += b[:, None, None]
    return out

  @staticmethod
  def conv2d_back(
    A: "Tensor", grad_out: np.ndarray, Weight: "Tensor", Bias: "Tensor",
    in_channels: int, out_channels: int, stride: int = 1, padding: int = 0
  ):
    a = A.data
    w = Weight.data
    b = Bias.data

    assert a.shape[1] == in_channels, "Input channels do not match"
    assert w.shape[0] == out_channels, "Output channels do not match"
    assert len(a.shape) == 4, "Input must be 4D (B, C, H, W)"
    assert len(w.shape) == 4, "Kernel must be 4D (C_out, C_in, H, W)"
    assert w.shape[2] == w.shape[3], "Kernel must be square"
    assert w.shape[1] % 2 == 1, "Kernel dimensions must be odd"
    assert a.shape[2] >= w.shape[1] and a.shape[3] >= w.shape[2], "Input must be larger than or equal to kernel dimensions"

    BS, C_in, H, W = a.shape
    C_out, _, kernel_size, _ = w.shape
    _, _, H_out, W_out = grad_out.shape

    grad_a = np.zeros_like(a)
    grad_w = np.zeros_like(w)
    grad_b = np.zeros_like(b)

    # Pad input and dA
    a_padded = np.pad(a, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
    grad_a_padded = np.pad(grad_a, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')

    for batch in range(BS):
      for out_c in range(C_out):
        for i in range(H_out):
          for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + kernel_size
            w_end = w_start + kernel_size

            grad_out_val = grad_out[batch, out_c, i, j]
            for in_c in range(C_in):
              input_patch = a_padded[batch, in_c, h_start:h_end, w_start:w_end]
              grad_w[out_c, in_c, :, :] += input_patch * grad_out_val # ∂L/∂W
              grad_a_padded[batch, in_c, h_start:h_end, w_start:w_end] += w[out_c, in_c, :, :] * grad_out_val # ∂L/∂A
            grad_b[out_c] += grad_out_val # ∂L/∂b

    # Remove padding from dA
    if padding > 0:
      grad_a = grad_a_padded[:, :, padding:-padding, padding:-padding]
    else:
      grad_a = grad_a_padded 

    if A.requires_grad: A.grad = grad_a
    if Weight.requires_grad: Weight.grad = grad_w
    if Bias.requires_grad: Bias.grad = grad_b


class UnaryOps:
  @staticmethod
  def abs(a: "Tensor") -> np.ndarray: pass
  
  @staticmethod
  def neg(a: "Tensor") -> np.ndarray: pass
  
  @staticmethod
  def sqrt(a: "Tensor") -> np.ndarray: pass
  
  @staticmethod
  def exp(a: "Tensor") -> np.ndarray: pass
  
  @staticmethod
  def log(a: "Tensor") -> np.ndarray: pass
  
  @staticmethod
  def normalize(a: "Tensor") -> np.ndarray: pass
  
  @staticmethod
  def batchnorm(a: np.ndarray) -> np.ndarray: pass

  @staticmethod
  def relu(a: "Tensor") -> np.ndarray: return np.maximum(a.data, np.zeros_like(a.data))

  @staticmethod
  def relu_back(a: "Tensor", grad_out: np.ndarray):
    if a.requires_grad: a.grad += np.where(a.data > 0, grad_out, 0)
  
  @staticmethod
  def sigmoid(a: "Tensor") -> np.ndarray: pass
  
  @staticmethod
  def tanh(a: "Tensor") -> np.ndarray: pass
  
  @staticmethod
  def softmax(a: "Tensor", axis: int) -> np.ndarray:
    exp_val = np.exp(a.data - np.max(a.data, axis=axis, keepdims=True))
    return exp_val / np.sum(exp_val, axis=axis, keepdims=True)

  @staticmethod
  def softmax_back(a: "Tensor", out: np.ndarray, grad_out: np.ndarray, axis: int):
    if not a.requires_grad: return
    dot = np.sum(grad_out * out, axis=axis, keepdims=True)
    grad_in = (grad_out - dot) * out
    a.grad += grad_in
    
  @staticmethod
  def tanh(a: "Tensor") -> np.ndarray: return np.tanh(a.data)

  @staticmethod
  def tanh_back(a: "Tensor", grad_out: np.ndarray):
    if a.requires_grad: a.grad += (1 - np.tanh(a.data)**2) * grad_out

  @staticmethod
  def sigmoid(a: "Tensor") -> np.ndarray: return np.exp(a.data) / (np.exp(a.data) + 1)

  @staticmethod
  def sigmoid_back(a: "Tensor", out: np.ndarray, grad_out: np.ndarray):
    if a.requires_grad: a.grad += out * (1 - out) * grad_out

class ReduceOps:
  @staticmethod
  def max(a: "Tensor", axis: int, keepdims: bool) -> np.ndarray:
    res = np.max(a.data, axis=axis, keepdims=keepdims)
    return res if keepdims else res[np.newaxis]

  @staticmethod
  def max_back(a: "Tensor", grad_out, axis: int, keepdims: bool):
    if not a.requires_grad: return
    if not keepdims and axis is not None:
      grad_out = np.expand_dims(grad_out, axis=axis)
    a.grad = (a.data == np.max(a.data, axis=axis, keepdims=True)) * grad_out

  @staticmethod
  def min(a: "Tensor", axis: int, keepdims: bool) -> np.ndarray:
    res = np.min(a.data, axis=axis, keepdims=keepdims)
    return res if keepdims else res[np.newaxis]

  @staticmethod
  def min_back(a: "Tensor", grad_out, axis: int, keepdims: bool):
    if not a.requires_grad: return
    if not keepdims and axis is not None:
      grad_out = np.expand_dims(grad_out, axis=axis)
    a.grad = (a.data == np.min(a.data, axis=axis, keepdims=True)) * grad_out

  @staticmethod
  def sum(a: "Tensor", axis: int, keepdims: bool) -> np.ndarray:
    res = np.sum(a.data, axis=axis, keepdims=keepdims)
    return res if keepdims else res[np.newaxis]

  @staticmethod
  def sum_back(a: "Tensor", grad_out, axis: int, keepdims: bool):
    if not a.requires_grad: return
    if not keepdims and axis is not None:
      grad_out = np.expand_dims(grad_out, axis=axis)
    a.grad = np.ones_like(a.data) * grad_out

  @staticmethod
  def mean(a: "Tensor", axis: int, keepdims: bool) -> np.ndarray:
    res  = np.mean(a.data, axis=axis, keepdims=keepdims)
    return res if keepdims else res[np.newaxis]

  @staticmethod
  def mean_back(a: "Tensor", grad_out, axis: int, keepdims: bool):
    if not a.requires_grad: return
    if not keepdims and axis is not None:
      grad_out = np.expand_dims(grad_out, axis=axis)
    a.grad = np.ones_like(a.data) * grad_out / a.data.size

  @staticmethod
  def std(a: "Tensor", axis: int, keepdims: bool) -> np.ndarray:
    res = np.std(a.data, axis=axis, keepdims=keepdims)
    return res if keepdims else res[np.newaxis]

  @staticmethod
  def std_back(a: "Tensor", grad_out, axis: int, keepdims: bool):
    if not a.requires_grad: return
    mean = np.mean(a.data, axis=axis, keepdims=True)
    std = np.std(a.data, axis=axis, keepdims=True)
    if not keepdims and axis is not None:
      grad_out = np.expand_dims(grad_out, axis=axis)
    a.grad = (a.data - mean) * grad_out / (std * a.data.size)

  @staticmethod
  def argmax(a: "Tensor", axis: int, keepdims: bool) -> np.ndarray:
    res = np.argmax(a.data, axis=axis, keepdims=keepdims)
    return res if keepdims else res[np.newaxis]

  @staticmethod
  def argmax_back(a: "Tensor"): a.grad = np.zeros_like(a.data)

  @staticmethod
  def argmin(a: "Tensor", axis: int, keepdims: bool) -> np.ndarray:
    res = np.argmin(a.data, axis=axis, keepdims=keepdims)
    return res if keepdims else res[np.newaxis]

  @staticmethod
  def argmin_back(a: "Tensor"): a.grad = np.zeros_like(a.data)

  @staticmethod
  def maxpool2d(a: "Tensor", filter=(2, 2), stride=1) -> np.ndarray:
    assert len(a.shape) == 4, "Input must be 3D (BS, C, H, W)"

    BS, channels, height, width = a.shape
    out_height = (height - filter[0]) // stride + 1
    out_width = (width - filter[1]) // stride + 1

    out = np.zeros((BS, channels, out_height, out_width))
    mask = np.zeros_like(a.data)
    for i in range(out_height):
      for j in range(out_width):
        h_start, h_end = i * stride, i * stride + filter[0]
        w_start, w_end = j * stride, j * stride + filter[1]

        x_slice = a.data[:, :, h_start:h_end, w_start:w_end]  # (BS, C, fh, fw)
        max_values = np.max(x_slice, axis=(2, 3), keepdims=True)  # keep dims for broadcasting

        out[:, :, i, j] = max_values[:, :, 0, 0]
        mask[:, :, h_start:h_end, w_start:w_end] = (x_slice == max_values)
    return out, mask

  @staticmethod
  def maxpool2d_back(a: "Tensor", grad_out: np.ndarray, mask: np.ndarray, filter: Tuple[int], stride: int):
    if not a.requires_grad: return

    BS, channels, height, width = a.shape
    out_height = (height - filter[0]) // stride + 1
    out_width = (width - filter[1]) // stride + 1

    grad_input = np.zeros_like(a.data)
    for i in range(out_height):
      for j in range(out_width):
        h_start, h_end = i * stride, i * stride + filter[0]
        w_start, w_end = j * stride, j * stride + filter[1]

        # broadcast grad_out over batch and channels
        grad_input[:, :, h_start:h_end, w_start:w_end] += mask[:, :, h_start:h_end, w_start:w_end] * grad_out[:, :, i, j][:, :, None, None]
    a.grad = grad_input if a.grad is None else a.grad + grad_input

  @staticmethod
  def avgpool2d(a: "Tensor", filter=(2, 2), stride=1) -> np.ndarray:
    # TODO: assert dimensionality
    # TODO: handle channels and padding as well
    # TODO: double-check if stride is used correctly
    assert len(a.shape) == 4, "Input must be 3D (BS, C, H, W)"

    BS, channels, height, width = a.shape
    out_height = (height - filter[0]) // stride + 1
    out_width = (width - filter[1]) // stride + 1

    out = np.zeros((BS, channels, out_height, out_width))
    for i in range(out_height):
      for j in range(out_width):
        h_start, h_end = i * stride, i * stride + filter[0]
        w_start, w_end = j * stride, j * stride + filter[1]

        x_slice = a.data[:, :, h_start:h_end, w_start:w_end]  # (BS, C, fh, fw)
        out[:, :, i, j] = np.mean(x_slice, axis=(2, 3))
    return out

  @staticmethod
  def avgpool2d_back(a: "Tensor", grad_out: np.ndarray, filter: Tuple[int], stride: int):
    if not a.requires_grad: return

    BS, channels, height, width = a.shape
    out_height = (height - filter[0]) // stride + 1
    out_width = (width - filter[1]) // stride + 1

    grad_input = np.zeros_like(a.data)
    for i in range(out_height):
      for j in range(out_width):
        h_start, h_end = i * stride, i * stride + filter[0]
        w_start, w_end = j * stride, j * stride + filter[1]

        grad_share = grad_out[:, :, i, j][:, :, None, None] / (filter[0] * filter[1])
        grad_input[:, :, h_start:h_end, w_start:w_end] += grad_share
    a.grad = grad_input if a.grad is None else a.grad + grad_input

  # loss functions

  @staticmethod
  def cross_entropy(z: "Tensor", y: "Tensor") -> np.ndarray:
    assert len(z.shape) == 2, "Z Tensor must be 2D (batch_size, num_classes)"
    assert len(y.shape) == 1, "Ground-truth Y must be 1D (batch_size,)"
    assert z.shape[0] == y.shape[0], "Z Tensor and ground-truth Y must have the same batch size"

    y.data = y.data.astype(np.int32)
    y_pred_clipped = np.clip(z.data, 1e-7, 1 - 1e-7)
    # loss_val = -np.sum(y.data * np.log(y_pred_clipped), axis=1)
    return -np.log(y_pred_clipped[np.arange(y.shape[0]), y.data])

  @staticmethod
  def cross_entropy_back(z: "Tensor", y: "Tensor") -> np.ndarray:
    if not z.requires_grad: return
    batch_size, n_classes = z.shape
    y_one_hot = np.zeros((batch_size, n_classes))
    y_one_hot[np.arange(batch_size), y.data] = 1
    z.grad = (z.data - y_one_hot) / batch_size  # Average gradient over batch


class MovementOps:
  @staticmethod
  def reshape(a: "Tensor", new_shape: Tuple[int]) -> np.ndarray: return a.data.reshape(new_shape)

  @staticmethod
  def reshape_back(a: "Tensor", grad_out: np.ndarray, original_shape: Tuple[int]):
    if a.requires_grad: a.grad += grad_out.reshape(original_shape)

  @staticmethod
  def transpose(a: "Tensor", axes: Tuple[int] = None) -> np.ndarray:
    if axes is None: axes = tuple(reversed(range(a.data.ndim)))
    return np.transpose(a.data, axes)

  @staticmethod
  def transpose_back(a: "Tensor", grad_out: np.ndarray, axes: Tuple[int] = None):
    if axes is None: axes = tuple(reversed(range(a.data.ndim)))
    if a.requires_grad:
      reverse_axes = np.argsort(axes)
      a.grad += np.transpose(grad_out, reverse_axes)

  @staticmethod
  def expand(a: "Tensor", new_shape: Tuple[int]) -> np.ndarray: return np.broadcast_to(a.data, new_shape)

  @staticmethod
  def expand_back(a: "Tensor", grad_out: np.ndarray, original_shape: Tuple[int]):
    if a.requires_grad: a.grad += reduce_grad(grad_out, original_shape)
  
  @staticmethod
  def permute(a: "Tensor", axes: Tuple[int]) -> np.ndarray: return np.transpose(a.data, axes)

  @staticmethod
  def permute_back(a: "Tensor", grad_out: np.ndarray, axes: Tuple[int]):
    if a.requires_grad:
      reverse_axes = np.argsort(axes)
      a.grad += np.transpose(grad_out, reverse_axes)

  @staticmethod
  def squeeze(a: "Tensor", axis: Tuple[int]) -> np.ndarray: return np.squeeze(a.data, axis=axis)
  
  @staticmethod
  def squeeze_back(a: "Tensor", grad_out: np.ndarray, axis: int, original_shape: Tuple[int]):
    if a.requires_grad:
      a.grad += np.expand_dims(grad_out, axis=axis).reshape(original_shape)

  @staticmethod
  def unsqueeze(a: "Tensor", axis: int) -> np.ndarray: return np.expand_dims(a.data, axis=axis)
  
  @staticmethod
  def unsqueeze_back(a: "Tensor", grad_out: np.ndarray, axis: int, original_shape: Tuple[int]):
    if a.requires_grad: a.grad = np.sum(grad_out, axis=axis).reshape(original_shape)
  

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

