import ctypes
import numpy as np
from typing import Tuple

from picograd.util import *
from .math import *

def reduce_grad(grad: np.ndarray, shape: tuple) -> np.ndarray:
  """Reduce broadcasted gradient `grad` back to `shape`."""

  # Step 1: collapse extra leading dims
  while grad.ndim > len(shape): grad = grad.sum(axis=0)

  # Step 2: sum over axes where original shape had 1 (broadcasted dims)
  for axis, dim in enumerate(shape):
    if dim == 1: grad = grad.sum(axis=axis, keepdims=True)

  return grad


class BinaryOps:
  @staticmethod
  def add(a: "Tensor", b: "Tensor", block_size: Tuple[int] = (8, 8, 8)) -> np.ndarray: # type: ignore
    """Add two homogeneous tensors of any dimension (1D, 2D, 3D) using CUDA."""
    kernel_code = a.device.manager.load_kernel("add.cu")
    kfunc = a.device.manager.compile_kernel(kernel_code, b"add_kernel")

    dims = a.shape
    padded_dims = dims + (1,) * (3 - len(dims))  # Pad to 3D
    dim1, dim2, dim3 = padded_dims[:3]

    C_flat = np.empty_like(a._data.ravel())
    d_C = a.device.manager.allocate_device_memory(C_flat)

    # Define grid and block sizes
    grid = (
      (dim3 + block_size[0] - 1) // block_size[0],
      (dim2 + block_size[1] - 1) // block_size[1],
      (dim1 + block_size[2] - 1) // block_size[2],
    )

    # Kernel launch and copy result back to host
    n_flops = dim1 * dim2 * dim3
    args = a.device.manager.prep_kargs(a.device_data, b.device_data, d_C, dim1, dim2, dim3)
    a.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops=n_flops)
    return d_C

  @staticmethod
  def add_back(a: "Tensor", b: "Tensor", grad_out: ctypes.c_void_p, block_size: Tuple[int] = (8, 8, 8)) -> np.ndarray:
    """Backward pass for addition operation."""

    if not a.requires_grad and not b.requires_grad: return

    kernel_code = a.device.manager.load_kernel("add.cu")
    kfunc = a.device.manager.compile_kernel(kernel_code, b"add_kernel")

    dims = a.shape
    padded_dims = dims + (1,) * (3 - len(dims))  # Pad to 3D
    dim1, dim2, dim3 = padded_dims[:3]

    grid = (
      (dim3 + block_size[0] - 1) // block_size[0],
      (dim2 + block_size[1] - 1) // block_size[1],
      (dim1 + block_size[2] - 1) // block_size[2],
    )

    n_flops = (int(a.requires_grad) + int(b.requires_grad)) * dim1 * dim2 * dim3
    if a.requires_grad:
      args = a.device.manager.prep_kargs(a.device_data, grad_out, a.device_grad, dim1, dim2, dim3)
      a.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops=n_flops)
    if b.requires_grad:
      args = a.device.manager.prep_kargs(b.device_data, grad_out, b.device_grad, dim1, dim2, dim3)
      a.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops=n_flops)

  @staticmethod
  def mul(a: "Tensor", b: "Tensor", block_size: Tuple[int] = (8, 8, 8)) -> np.ndarray:
    """Pointwise multiplication"""
    assert a.shape == b.shape, "Tensors must have the same shape"

    kernel_code = a.device.manager.load_kernel("mul.cu")
    kfunc = a.device.manager.compile_kernel(kernel_code, b"mul_kernel")

    dims = a.shape
    padded_dims = dims + (1,) * (3 - len(dims))  # Pad to 3D
    dim1, dim2, dim3 = padded_dims[:3]

    C_flat = np.empty_like(a._data.ravel())
    d_C = a.device.manager.allocate_device_memory(C_flat)

    # Define grid and block sizes
    grid = (
      (dim3 + block_size[0] - 1) // block_size[0],
      (dim2 + block_size[1] - 1) // block_size[1],
      (dim1 + block_size[2] - 1) // block_size[2],
    )

    # Kernel launch and copy result back to host
    n_flops = dim1 * dim2 * dim3
    args = a.device.manager.prep_kargs(a.device_data, b.device_data, d_C, dim1, dim2, dim3)
    a.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops=n_flops)
    return d_C

  @staticmethod
  def mul_back(a: "Tensor", b: "Tensor", grad_out: ctypes.c_void_p,  block_size=(8, 8, 8)) -> np.ndarray:
    """Backward pass for pointwise multiplication operation."""
    if not a.requires_grad and not b.requires_grad: return

    add_kernel_code = a.device.manager.load_kernel("add.cu")
    add_kfunc = a.device.manager.compile_kernel(add_kernel_code, b"add_kernel")
    mul_kernel_code = a.device.manager.load_kernel("mul.cu")
    mul_kfunc = a.device.manager.compile_kernel(mul_kernel_code, b"mul_kernel")

    dims = a.shape
    padded_dims = dims + (1,) * (3 - len(dims))  # Pad to 3D
    dim1, dim2, dim3 = padded_dims[:3]

    # for storing the mul result, to be used for addition
    temp_a = np.zeros_like(a._grad.ravel())
    temp_b = np.zeros_like(b._grad.ravel())

    _, d_temp_a = a.device.manager.np_to_device(temp_a)
    _, d_temp_b = a.device.manager.np_to_device(temp_b)

    grid = (
      (dim3 + block_size[0] - 1) // block_size[0],
      (dim2 + block_size[1] - 1) // block_size[1],
      (dim1 + block_size[2] - 1) // block_size[2],
    )
    n_flops = (int(a.requires_grad) + int(b.requires_grad)) * dim1 * dim2 * dim3

    if a.requires_grad:
      # d_temp_a = b.device_data * grad_out
      kargs = a.device.manager.prep_kargs(b.device_data, grad_out, d_temp_a, dim1, dim2, dim3)
      a.device.manager.launch_kernel(mul_kfunc, grid, block_size, kargs, n_flops=n_flops)

      # a.device_grad += d_temp_a
      kargs = a.device.manager.prep_kargs(a.device_grad, d_temp_a, a.device_grad, dim1, dim2, dim3)
      a.device.manager.launch_kernel(add_kfunc, grid, block_size, kargs, n_flops=n_flops)

      # a.device.manager.free_device_tensor(d_temp_a)  # FIXME: segfaults

    if b.requires_grad:
      # d_temp_b = a.device_data * grad_out
      kargs = a.device.manager.prep_kargs(a.device_data, grad_out, d_temp_b, dim1, dim2, dim3)
      a.device.manager.launch_kernel(mul_kfunc, grid, block_size, kargs, n_flops=n_flops)

      # b.device_grad += d_temp_b
      kargs = a.device.manager.prep_kargs(b.device_grad, d_temp_b, b.device_grad, dim1, dim2, dim3)
      a.device.manager.launch_kernel(add_kfunc, grid, block_size, kargs, n_flops=n_flops)

      # free_device_tensor(a.device.manager, d_temp_b)  # FIXME: segfaults

  @staticmethod
  def dot(a: "Tensor", b: "Tensor") -> np.ndarray:
    """Matrix multiplication using CUDA."""
    assert len(a.shape) == 2 and len(b.shape) == 2, "Both tensors must be 2D (matrices)"
    assert a.shape[1] == b.shape[0], "Inner dimensions must match"

    kernel_code = a.device.manager.load_kernel("matmul.cu")
    kfunc = a.device.manager.compile_kernel(kernel_code, b"matmul_tiled_kernel")

    M, K = a.shape
    _, N = b.shape[0], b.shape[1]

    C = np.zeros((M, N), dtype=np.float32)
    d_C = a.device.manager.allocate_device_memory(C)

    block_size = (a.device.manager.tile_size, a.device.manager.tile_size, 1)
    grid = (
      (N + a.device.manager.tile_size - 1) // a.device.manager.tile_size,
      (M + a.device.manager.tile_size - 1) // a.device.manager.tile_size,
      1,
    )

    num_flops = 2 * M * N * K
    args = a.device.manager.prep_kargs(a.device_data, b.device_data, d_C, M, N, K)
    a.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops=num_flops)
    return d_C

  @staticmethod
  def dot_back(a: "Tensor", b: "Tensor", grad_out: ctypes.c_void_p) -> np.ndarray:
    if not a.requires_grad and not b.requires_grad: return

    assert len(a.shape) == 2 and len(b.shape) == 2, "Both tensors must be 2D (matrices)"
    assert a.shape[1] == b.shape[0], "Inner dimensions must match"

    dot_kernel_code = a.device.manager.load_kernel("matmul.cu")
    dot_kfunc = a.device.manager.compile_kernel(dot_kernel_code, b"matmul_tiled_kernel")
    add_kernel_code = a.device.manager.load_kernel("add.cu")
    add_kfunc = a.device.manager.compile_kernel(add_kernel_code, b"add_kernel")

    temp_a = np.zeros_like(a._grad.ravel())
    temp_b = np.zeros_like(b._grad.ravel())
    
    _, d_temp_a = a.device.manager.np_to_device(temp_a)
    _, d_temp_b = a.device.manager.np_to_device(temp_b)

    # setup dot kernel
    M, K = a.shape
    _, N = b.shape[0], b.shape[1]
    dot_block_size = (a.device.manager.tile_size, a.device.manager.tile_size, 1)
    dot_grid = (
      (N + a.device.manager.tile_size - 1) // a.device.manager.tile_size,
      (M + a.device.manager.tile_size - 1) // a.device.manager.tile_size,
      1,
    )

    # setup add kernel
    dims = a.shape
    padded_dims = dims + (1,) * (3 - len(dims))  # Pad to 3D
    dim1, dim2, dim3 = padded_dims[:3]
    add_block_size = (8, 8, 8)
    add_grid = (
      (dim3 + add_block_size[0] - 1) // add_block_size[0],
      (dim2 + add_block_size[1] - 1) // add_block_size[1],
      (dim1 + add_block_size[2] - 1) // add_block_size[2],
    )

    num_flops = 2 * M * N * K

    # TODO: transpose b and a ?
    if a.requires_grad:
      # d_temp_a = grad_out @ b.data.T
      args = a.device.manager.prep_kargs(grad_out, b.device_data, d_temp_a, M, N, K)
      a.device.manager.launch_kernel(dot_kfunc, dot_grid, dot_block_size, args, n_flops=num_flops)

      # a.device_grad += d_temp_a
      args = a.device.manager.prep_kargs(a.device_grad, d_temp_a, a.device_grad, dim1, dim2, dim3)
      a.device.manager.launch_kernel(add_kfunc, add_grid, add_block_size, args, n_flops=num_flops)

      # a.device.manager.free_device_tensor(d_temp_a)  # FIXME: segfaults

    if b.requires_grad:
      # d_temp_b = a.data.T @ grad_out
      args = a.device.manager.prep_kargs(a.device_data, grad_out, d_temp_b, K, M, N)
      a.device.manager.launch_kernel(dot_kfunc, dot_grid, dot_block_size, args, n_flops=num_flops)

      # b.device_grad += d_temp_b
      args = a.device.manager.prep_kargs(b.device_grad, d_temp_b, b.device_grad, dim1, dim2, dim3)
      a.device.manager.launch_kernel(add_kfunc, add_grid, add_block_size, args, n_flops=num_flops)

      # a.device.manager.free_device_tensor(d_temp_b)  # FIXME: segfaults

  # TODO: this is naive conv2d
  @staticmethod
  def conv2d(
    a: "Tensor", w: "Tensor", b:"Tensor",
    in_channels: int, out_channels: int, stride: int = 1, padding: int = 0,
    block_size: Tuple[int] = (256, 1, 1)
  ) -> np.ndarray:
    assert len(a.shape) == 4, "Input must be 4D (B, C, H, W)"
    assert a.shape[1] == in_channels, "Input channels do not match"
    assert len(w.shape) == 4, "Kernel must be 4D (C_out, C_in, H, W)"
    assert w.shape[2] == w.shape[3], "Kernel must be square"
    assert w.shape[1] % 2 == 1, "Kernel dimensions must be odd"
    assert a.shape[2] >= w.shape[1] and a.shape[3] >= w.shape[2], "Input must be larger than or equal to kernel dimensions"

    kernel_code = a.device.manager.load_kernel("conv2d.cu")
    kfunc = a.device.manager.compile_kernel(kernel_code, b"conv2d_kernel")

    BS, C_in, H, W = a.shape
    C_out, C_in, kernel_size, _ = w.shape
    H_out = ((H - kernel_size + 2*padding) // stride) + 1
    W_out = ((W - kernel_size + 2*padding) // stride) + 1

    # FIXME: a_padded

    C = np.zeros((BS, out_channels, H_out, W_out))
    d_C = a.device.manager.allocate_device_memory(C)

    grid = (BS, out_channels, 4)
    num_flops = BS * out_channels * H_out * W_out * in_channels * kernel_size * kernel_size * 2
    args = a.device.manager.prep_kargs(
      a.device_data, w.device_data, b.device_data, d_C,
      BS, in_channels, H, W,
      out_channels, kernel_size, kernel_size,
      H_out, W_out,
      stride, padding
    )
    a.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops=num_flops)
    return d_C

  @staticmethod
  def conv2d_back(
    a: "Tensor", grad_out: ctypes.c_void_p, w: "Tensor", b: "Tensor",
    in_channels: int, out_channels: int, stride: int = 1, padding: int = 0,
    block_size: Tuple[int] = (256, 1, 1)
  ):
    if not a.requires_grad and not w.requires_grad and not b.requires_grad: return

    assert a.shape[1] == in_channels, "Input channels do not match"
    assert w.shape[0] == out_channels, "Output channels do not match"
    assert len(a.shape) == 4, "Input must be 4D (B, C, H, W)"
    assert len(w.shape) == 4, "Kernel must be 4D (C_out, C_in, H, W)"
    assert w.shape[2] == w.shape[3], "Kernel must be square"
    assert w.shape[1] % 2 == 1, "Kernel dimensions must be odd"
    assert a.shape[2] >= w.shape[1] and a.shape[3] >= w.shape[2], "Input must be larger than or equal to kernel dimensions"

    kernel_code = a.device.manager.load_kernel("conv2d_back.cu")
    kfunc = a.device.manager.compile_kernel(kernel_code, b"conv2d_backward_kernel")

    BS, C_in, H, W = a.shape
    C_out, C_in, kernel_size, _ = w.shape
    H_out = ((H - kernel_size + 2*padding) // stride) + 1
    W_out = ((W - kernel_size + 2*padding) // stride) + 1

    grad_a = np.zeros_like(a)
    d_grad_a = a.device.manager.allocate_device_memory(grad_a)
    grad_w = np.zeros_like(w)
    d_grad_w = a.device.manager.allocate_device_memory(grad_w)
    grad_b = np.zeros_like(b)
    d_grad_b = a.device.manager.allocate_device_memory(grad_b)

    # FIXME: a_padded + grad_a_padded

    grid = (BS, out_channels, 4)
    num_flops = BS * out_channels * H_out * W_out * in_channels * kernel_size * kernel_size * 2
    args = a.device.manager.prep_kargs(
      a.device_data, w.device_data,
      grad_out, d_grad_a, d_grad_w, d_grad_b,
      BS, in_channels, out_channels,
      H, W,
      H_out, W_out,
      H_out, W_out,
      kernel_size,
      stride,
      padding
    )
    a.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops=num_flops)

    if a.requires_grad: a.device_grad = d_grad_a
    if w.requires_grad: w.device_grad = d_grad_w
    if b.requires_grad: b.device_grad = d_grad_b

class UnaryOps:
  @staticmethod
  def relu(a: "Tensor", block_size: Tuple[int] = (256, 1, 1)) -> np.ndarray:
    kernel_code = a.device.manager.load_kernel("relu.cu")
    kfunc = a.device.manager.compile_kernel(kernel_code, b"relu_kernel")

    size = int(np.prod(a.shape))
    C_flat = np.zeros(size, dtype=np.float32)
    d_C = a.device.manager.allocate_device_memory(C_flat)

    grid = ((size + block_size[0] - 1) // block_size[0], 1, 1)
    args = a.device.manager.prep_kargs(a.device_data, d_C, size)
    a.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops=size)
    return d_C

  # FIXME: [CUDA ERROR] cuMemcpyDtoH failed: CUDA_ERROR_ILLEGAL_ADDRESS (code 700) on MNIST after a few iterations
  @staticmethod
  def relu_back(a: "Tensor", grad_out: ctypes.c_void_p, block_size: Tuple[int] = (256, 1, 1)):
    if not a.requires_grad: return

    kernel_code = a.device.manager.load_kernel("relu_back.cu")
    kfunc = a.device.manager.compile_kernel(kernel_code, b"relu_back_kernel")

    size = int(np.prod(a.shape))

    grid = ((size + block_size[0] - 1) // block_size[0], 1, 1)
    args = a.device.manager.prep_kargs(a.device_data, grad_out, a.device_grad, size)
    a.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops=size)

  @staticmethod
  def softmax(a: "Tensor", block_size: Tuple[int] = (256, 1, 1)) -> np.ndarray:
    assert len(a.shape) == 2, "Softmax is only implemented for 2D tensors (batch_size, num_classes)"

    kernel_code = a.device.manager.load_kernel("softmax.cu")
    kfunc = a.device.manager.compile_kernel(kernel_code, b"softmax_kernel")

    batch_size, n_classes = a.shape
    size = batch_size * n_classes

    C_flat = np.zeros(size, dtype=np.float32)
    d_C = a.device.manager.allocate_device_memory(C_flat)

    grid = (batch_size, 1, 1)
    shared_mem = block_size[0] * np.dtype(np.float32).itemsize
    n_flops = 3 * size
    args = a.device.manager.prep_kargs(a.device_data, d_C, batch_size, n_classes)
    a.device.manager.launch_kernel(kfunc, grid, block_size, args, shared_mem=shared_mem, n_flops=n_flops)
    return d_C

  @staticmethod
  def softmax_back(a: "Tensor", softmax_out: ctypes.c_void_p, grad_out: ctypes.c_void_p, block_size: Tuple[int] = (256, 1, 1)):
    if not a.requires_grad: return

    kernel_code = a.device.manager.load_kernel("softmax_back.cu")
    kfunc = a.device.manager.compile_kernel(kernel_code, b"softmax_back_kernel")

    batch_size, n_classes = a.shape
    size = batch_size * n_classes

    grid = (batch_size, 1, 1)
    shared_mem = block_size[0] * np.dtype(np.float32).itemsize
    n_flops = 3 * size
    args = a.device.manager.prep_kargs(
      grad_out,
      softmax_out,
      a.device_grad,
      batch_size,
      n_classes
    )
    a.device.manager.launch_kernel(kfunc, grid, block_size, args, shared_mem=shared_mem, n_flops=n_flops)

  @staticmethod
  def sigmoid(a: "Tensor") -> np.ndarray: raise NotImplementedError("This op is not implemented yet")
  
  @staticmethod
  def tanh(a: "Tensor") -> np.ndarray: raise NotImplementedError("This op is not implemented yet")
  
  @staticmethod
  def abs(a: "Tensor") -> np.ndarray: raise NotImplementedError("This op is not implemented yet")
  
  @staticmethod
  def neg(a: "Tensor") -> np.ndarray: raise NotImplementedError("This op is not implemented yet")
  
  @staticmethod
  def sqrt(a: "Tensor") -> np.ndarray: raise NotImplementedError("This op is not implemented yet")
  
  @staticmethod
  def exp(a: "Tensor") -> np.ndarray: raise NotImplementedError("This op is not implemented yet")
  
  @staticmethod
  def log(a: "Tensor") -> np.ndarray: raise NotImplementedError("This op is not implemented yet")
  
  @staticmethod
  def normalize(a: "Tensor") -> np.ndarray: raise NotImplementedError("This op is not implemented yet")
  
  @staticmethod
  def batchnorm(a: "Tensor") -> np.ndarray: raise NotImplementedError("This op is not implemented yet")


class ReduceOps:
  @staticmethod
  def maxpool2d(a: "Tensor", filter=(2, 2), stride=1) -> np.ndarray: raise NotImplementedError("This op is not implemented yet")

  @staticmethod
  def maxpool2d_back(a: "Tensor", grad_out: ctypes.c_void_p, mask: np.ndarray, filter: Tuple[int], stride: int): raise NotImplementedError("This op is not implemented yet")

  @staticmethod
  def avgpool2d(a: "Tensor", filter=(2, 2), stride=1) -> np.ndarray: raise NotImplementedError("This op is not implemented yet")

  @staticmethod
  def avgpool2d_back(a: "Tensor", grad_out: ctypes.c_void_p, filter: Tuple[int], stride: int): raise NotImplementedError("This op is not implemented yet")

  # loss functions

  @staticmethod
  def cross_entropy(z: "Tensor", y: "Tensor", block_size=(256, 1, 1)) -> np.ndarray:
    assert len(z.shape) == 2, "Z Tensor must be 2D (batch_size, num_classes)"
    assert len(y.shape) == 1, "Ground-truth Y must be 1D (batch_size,)"
    assert z.shape[0] == y.shape[0], "Z Tensor and ground-truth Y must have the same batch size"

    kernel_code = z.device.manager.load_kernel("cre.cu")
    kfunc = z.device.manager.compile_kernel(kernel_code, b"cross_entropy_kernel")
    
    batch_size, n_classes = z.shape
    loss_flat = np.zeros(batch_size, dtype=np.float32)
    d_loss = z.device.manager.allocate_device_memory(loss_flat)

    grid = ((batch_size + block_size[0] - 1) // block_size[0], 1, 1)
    args = z.device.manager.prep_kargs(z.device_data, y.device_data, d_loss, batch_size, n_classes)
    z.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops=batch_size * n_classes)
    return d_loss

  @staticmethod
  def cross_entropy_back(z: "Tensor", y: "Tensor", block_size=(256, 1, 1)):
    if not z.requires_grad: return

    kernel_code = z.device.manager.load_kernel("cre_back.cu")
    kfunc = z.device.manager.compile_kernel(kernel_code, b"cross_entropy_back_kernel")

    batch_size, n_classes = z.shape
    grid = ((batch_size + block_size[0] - 1) // block_size[0], 1, 1)
    args = z.device.manager.prep_kargs(z.device_data, y.device_data, z.device_grad, batch_size, n_classes)
    z.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops=batch_size * n_classes)


# TODO: implement shapetracker to remove numpy from here
class MovementOps:
  @staticmethod
  def reshape(a: "Tensor", new_shape: Tuple[int]) -> np.ndarray: return a.device_data

  @staticmethod
  def reshape_back(a: "Tensor", grad_out: np.ndarray, original_shape: Tuple[int]):
    if a.requires_grad: a.device_grad = cuda_add_gpu(a.device_grad, grad_out, a.device.manager, original_shape)

  @staticmethod
  def transpose(a: "Tensor", axes: Tuple[int] = None) -> np.ndarray: return a.device_data

  # TODO: implement transpose_back
  @staticmethod
  def transpose_back(a: "Tensor", grad_out: np.ndarray, axes: Tuple[int] = None):
    if a.requires_grad: return None

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
