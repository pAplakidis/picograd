import numpy as np
from enum import Enum, auto
from typing import Tuple

from picograd.util import *
from .utils import *


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


class BinaryOps:
  @staticmethod
  def add(a: "Tensor", b: "Tensor", block_size: Tuple = (8, 8, 8)) -> np.ndarray: # type: ignore
    """Add two homogeneous tensors of any dimension (1D, 2D, 3D) using CUDA."""
    assert a.shape == b.shape, "Tensors must have the same shape"

    # TODO: check if kernel is already loaded (use manager.kernels dict)
    kernel_code = a.device.manager.load_kernel("add.cu")
    a.device.manager.compile_kernel(kernel_code, b"add_kernel")

    dims = a.shape
    padded_dims = dims + (1,) * (3 - len(dims))  # Pad to 3D
    dim1, dim2, dim3 = padded_dims[:3]

    C_flat = np.empty_like(a.data.ravel())
    d_C = allocate_device_memory(a.device.manager, C_flat)

    # TODO: double check and study this
    # Define grid and block sizes
    grid = (
      (dim3 + block_size[0] - 1) // block_size[0],
      (dim2 + block_size[1] - 1) // block_size[1],
      (dim1 + block_size[2] - 1) // block_size[2],
    )

    # Kernel launch and copy result back to host
    args = prep_kargs(a.device_data, b.device_data, d_C, dim1, dim2, dim3)
    a.device.manager.launch_kernel(a.device.manager.kfunc, grid, block_size, args)
    a.device.manager.memcpy_dtoh(C_flat.ctypes.data, d_C, C_flat.nbytes)

    # TODO: these belong in tensor.to(cpu)
    # free_device_tensor(manager, a.device_data)
    # free_device_tensor(manager, b.device_data)
    # free_device_tensor(manager, d_C)

    return C_flat.reshape(dims)

  @staticmethod
  def mul(a: "Tensor", b: "Tensor", block_size: Tuple = (8, 8, 8)) -> np.ndarray:
    assert a.shape == b.shape, "Tensors must have the same shape"

    kernel_code = a.device.manager.load_kernel("mul.cu")
    a.device.manager.compile_kernel(kernel_code, b"mul_kernel")

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
    args = prep_kargs(a.device_data, b.device_data, d_C, dim1, dim2, dim3)
    a.device.manager.launch_kernel(a.device.manager.kfunc, grid, block_size, args)
    a.device.manager.memcpy_dtoh(C_flat.ctypes.data, d_C, C_flat.nbytes)

    # TODO: these belong in tensor.to(cpu)
    # free_device_tensor(manager, a.device_data)
    # free_device_tensor(manager, b.device_data)
    # free_device_tensor(manager, d_C)

    return C_flat.reshape(dims)

  @staticmethod
  def dot(a: "Tensor", b: "Tensor") -> np.ndarray: raise NotImplementedError("BinaryOps.dot is not implemented yet")

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
  def sigmoid(a: "Tensor") -> np.ndarray: pass
  
  @staticmethod
  def tanh(a: "Tensor") -> np.ndarray: pass
  
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
  def softmax(a: "Tensor") -> np.ndarray: pass
  
  @staticmethod
  def batchnorm(a: "Tensor") -> np.ndarray: pass
