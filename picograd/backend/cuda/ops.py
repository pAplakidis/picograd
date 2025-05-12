import numpy as np
from enum import Enum, auto
from typing import Tuple

from picograd.util import *
from .utils import *
from .math import *


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


# TODO: backward ops should keep gradients in the device memory
class BinaryOps:
  @staticmethod
  def add(a: "Tensor", b: "Tensor", block_size: Tuple = (8, 8, 8)) -> np.ndarray: # type: ignore
    """Add two homogeneous tensors of any dimension (1D, 2D, 3D) using CUDA."""
    kernel_code = a.device.manager.load_kernel("add.cu")
    kfunc = a.device.manager.compile_kernel(kernel_code, b"add_kernel")

    dims = a.shape
    padded_dims = dims + (1,) * (3 - len(dims))  # Pad to 3D
    dim1, dim2, dim3 = padded_dims[:3]

    C_flat = np.empty_like(a.data.ravel())
    d_C = allocate_device_memory(a.device.manager, C_flat)

    # Define grid and block sizes
    grid = (
      (dim3 + block_size[0] - 1) // block_size[0],
      (dim2 + block_size[1] - 1) // block_size[1],
      (dim1 + block_size[2] - 1) // block_size[2],
    )

    # Kernel launch and copy result back to host
    n_flops = dim1 * dim2 * dim3
    args = prep_kargs(a.device_data, b.device_data, d_C, dim1, dim2, dim3)
    a.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops)
    a.device.manager.memcpy_dtoh(C_flat.ctypes.data, d_C, C_flat.nbytes)

    return C_flat.reshape(dims), d_C

  @staticmethod
  def add_back(a: "Tensor", b: "Tensor", grad_out: np.ndarray) -> np.ndarray:
    if a.requires_grad: a.grad = cuda_add(a.data, grad_out, a.device.manager)
    if b.requires_grad:
      if b.grad.shape != grad_out.shape:
        b.grad = cuda_add(b.grad, np.sum(grad_out, axis=0), b.device.manager) # TODO: move sum to CUDA (?)
      else:
        b.grad = cuda_add(b.grad, grad_out, b.device.manager)

  @staticmethod
  def mul(a: "Tensor", b: "Tensor", block_size: Tuple = (8, 8, 8)) -> np.ndarray:
    assert a.shape == b.shape, "Tensors must have the same shape"

    kernel_code = a.device.manager.load_kernel("mul.cu")
    kfunc = a.device.manager.compile_kernel(kernel_code, b"mul_kernel")

    dims = a.shape
    padded_dims = dims + (1,) * (3 - len(dims))  # Pad to 3D
    dim1, dim2, dim3 = padded_dims[:3]

    C_flat = np.empty_like(a.data.ravel())
    d_C = allocate_device_memory(a.device.manager, C_flat)

    # Define grid and block sizes
    grid = (
      (dim3 + block_size[0] - 1) // block_size[0],
      (dim2 + block_size[1] - 1) // block_size[1],
      (dim1 + block_size[2] - 1) // block_size[2],
    )

    # Kernel launch and copy result back to host
    n_flops = dim1 * dim2 * dim3
    args = prep_kargs(a.device_data, b.device_data, d_C, dim1, dim2, dim3)
    a.device.manager.launch_kernel(kfunc, grid, block_size, args, n_flops)
    a.device.manager.memcpy_dtoh(C_flat.ctypes.data, d_C, C_flat.nbytes)

    return C_flat.reshape(dims), d_C

  @staticmethod
  def mul_back(a: "Tensor", b: "Tensor", grad_out: np.ndarray) -> np.ndarray:
    if a.requires_grad: a.grad = cuda_add(a.grad, cuda_mul(b.data, grad_out, a.device.manager), a.device.manager)
    if b.requires_grad: b.grad += a.data * grad_out
    if b.requires_grad: b.grad = cuda_add(b.grad, cuda_mul(a.data, grad_out, b.device.manager), b.device.manager)

  @staticmethod
  def dot(a: "Tensor", b: "Tensor", block_size: Tuple = (8, 8, 1)) -> np.ndarray:
    """Matrix multiplication using CUDA."""
    assert a.shape[1] == b.shape[0], "Inner dimensions must match"

    kernel_code = a.device.manager.load_kernel("matmul.cu")
    kfunc = a.device.manager.compile_kernel(kernel_code, b"matmul_tiled_kernel")

    M, K = a.shape
    _, N = b.shape[0], b.shape[1]

    C = np.zeros((M, N), dtype=np.float32)
    d_C = allocate_device_memory(a.device.manager, C)

    grid = (
      (N + TILE_SIZE - 1) // TILE_SIZE,
      (M + TILE_SIZE - 1) // TILE_SIZE,
      1,
    )
    block_size = (TILE_SIZE, TILE_SIZE, 1)

    num_flops = 2 * M * N * K
    args = prep_kargs(a.device_data, b.device_data, d_C, M, N, K)
    a.device.manager.launch_kernel(kfunc, grid, block_size, args, num_flops)
    a.device.manager.memcpy_dtoh(C.ctypes.data, d_C, C.nbytes)

    return C, d_C

  @staticmethod
  def dot_back(a: "Tensor", b: "Tensor", grad_out: np.ndarray) -> np.ndarray:
    if a.requires_grad: a.grad = cuda_add(a.grad, cuda_gemm(grad_out, b.data.T, a.device.manager), a.device.manager)
    if b.requires_grad: b.grad = cuda_add(b.grad, cuda_gemm(a.data.T, grad_out, b.device.manager), b.device.manager)

  @staticmethod
  def conv2d(a: "Tensor", w: "Tensor", b:"Tensor",
             in_channels: int, out_channels: int, stride: int = 1, padding: int = 0,
             debug=False) -> np.ndarray:
    raise NotImplementedError("BinaryOps.conv2d is not implemented yet")

  @staticmethod
  def conv2d_backward(a: "Tensor", grad_out: np.ndarray, w: "Tensor", b: "Tensor",
                      in_channels: int, out_channels: int, stride: int = 1, padding: int = 0):
    raise NotImplementedError("BinaryOps.conv2d_backward is not implemented yet")

class UnaryOps:
  @staticmethod
  def relu(a: "Tensor") -> np.ndarray: return np.maximum(a, np.zeros_like(a))
  
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
  def softmax(a: "Tensor") -> np.ndarray: raise NotImplementedError("This op is not implemented yet")
  
  @staticmethod
  def batchnorm(a: "Tensor") -> np.ndarray: raise NotImplementedError("This op is not implemented yet")
