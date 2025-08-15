import numpy as np
from typing import Optional, Tuple, List

from picograd.backend.cuda.cuda import CudaDeviceManager

# Generic math operations for CUDA, using NumPy arrays

# FIXME: no need to allocate device memeory for gradients (creates memory leaks)
def cuda_add(A: np.ndarray, B: np.ndarray, dev_manager: CudaDeviceManager, block_size: Tuple = (8, 8, 8)) -> np.ndarray:
  """CUDA - Add two 1D, 2D or 3D arrays."""
  assert A.shape == B.shape, "Tensors must have the same shape"

  kernel_code = dev_manager.load_kernel("add.cu")
  kfunc = dev_manager.compile_kernel(kernel_code, b"add_kernel")

  dims = A.shape
  padded_dims = dims + (1,) * (3 - len(dims))  # Pad to 3D
  dim1, dim2, dim3 = padded_dims[:3]

  A_flat, B_flat = A.ravel(), B.ravel()
  C_flat = np.empty_like(A_flat)
  d_A = dev_manager.allocate_device_memory(dev_manager, A_flat)
  d_B = dev_manager.allocate_device_memory(dev_manager, B_flat)
  d_C = dev_manager.allocate_device_memory(dev_manager, C_flat)
  dev_manager.copy_data_to_device(dev_manager, d_A, A_flat)
  dev_manager.copy_data_to_device(dev_manager, d_B, B_flat)

  # Define grid and block sizes
  grid = (
    (dim3 + block_size[0] - 1) // block_size[0],
    (dim2 + block_size[1] - 1) // block_size[1],
    (dim1 + block_size[2] - 1) // block_size[2],
  )

  # Kernel launch and copy result back to host
  args = dev_manager.prep_kargs(d_A, d_B, d_C, dim1, dim2, dim3)
  dev_manager.launch_kernel(kfunc, grid, block_size, args)
  dev_manager.memcpy_dtoh(C_flat.ctypes.data, d_C, C_flat.nbytes)

  dev_manager.free_device_tensor(dev_manager, d_A)
  dev_manager.free_device_tensor(dev_manager, d_B)
  dev_manager.free_device_tensor(dev_manager, d_C)

  return C_flat.reshape(dims)

def cuda_mul(A: np.ndarray, B: np.ndarray, dev_manager: CudaDeviceManager, block_size: Tuple = (8, 8, 8)) -> np.ndarray:
  """CUDA - Pointwise multiply two 1D, 2D or 3D arrays."""
  assert A.shape == B.shape, "Tensors must have the same shape"

  kernel_code = dev_manager.load_kernel("mul.cu")
  kfunc = dev_manager.compile_kernel(kernel_code, b"mul_kernel")

  dims = A.shape
  padded_dims = dims + (1,) * (3 - len(dims))  # Pad to 3D
  dim1, dim2, dim3 = padded_dims[:3]

  A_flat, B_flat = A.ravel(), B.ravel()
  C_flat = np.empty_like(A_flat)
  d_A = dev_manager.allocate_device_memory(dev_manager, A_flat)
  d_B = dev_manager.allocate_device_memory(dev_manager, B_flat)
  d_C = dev_manager.allocate_device_memory(dev_manager, C_flat)
  dev_manager.copy_data_to_device(dev_manager, d_A, A_flat)
  dev_manager.copy_data_to_device(dev_manager, d_B, B_flat)

  # Define grid and block sizes
  grid = (
    (dim3 + block_size[0] - 1) // block_size[0],
    (dim2 + block_size[1] - 1) // block_size[1],
    (dim1 + block_size[2] - 1) // block_size[2],
  )

  # Kernel launch and copy result back to host
  args = dev_manager.prep_kargs(d_A, d_B, d_C, dim1, dim2, dim3)
  dev_manager.launch_kernel(kfunc, grid, block_size, args)
  dev_manager.memcpy_dtoh(C_flat.ctypes.data, d_C, C_flat.nbytes)

  dev_manager.free_device_tensor(dev_manager, d_A)
  dev_manager.free_device_tensor(dev_manager, d_B)
  dev_manager.free_device_tensor(dev_manager, d_C)

  return C_flat.reshape(dims)

def cuda_gemm(A: np.ndarray, B: np.ndarray, dev_manager: CudaDeviceManager, block_size: Tuple = (8, 8, 1), tile_size: int = 16) -> np.ndarray:
  """CUDA - General matrix multiplication (GEMM) for 2D arrays."""
  assert A.shape[1] == B.shape[0], "Inner dimensions must match"

  kernel_code = dev_manager.load_kernel("matmul.cu")
  kfunc = dev_manager.compile_kernel(kernel_code, b"matmul_tiled_kernel")

  M, K = A.shape
  _, N = B.shape[1], B.shape[1]

  C = np.zeros((M, N), dtype=np.float32)
  d_A = dev_manager.allocate_device_memory(dev_manager, A)
  d_B = dev_manager.allocate_device_memory(dev_manager, B)
  d_C = dev_manager.allocate_device_memory(dev_manager, C)
  dev_manager.copy_data_to_device(dev_manager, d_A, A)
  dev_manager.copy_data_to_device(dev_manager, d_B, B)

  # Define grid and block sizes
  grid = (
    (N + tile_size - 1) // tile_size,
    (M + tile_size - 1) // tile_size,
    1,
  )
  block_size = (tile_size, tile_size, 1)
  
  # Kernel launch and copy result back to host
  args = dev_manager.prep_kargs(d_A, d_B, d_C, M, N, K)
  dev_manager.launch_kernel(kfunc, grid, block_size, args)
  dev_manager.memcpy_dtoh(C.ctypes.data, d_C, C.nbytes)

  dev_manager.free_device_tensor(dev_manager, d_A)
  dev_manager.free_device_tensor(dev_manager, d_B)
  dev_manager.free_device_tensor(dev_manager, d_C)

  return C
