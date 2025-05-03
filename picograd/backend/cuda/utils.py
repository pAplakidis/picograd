import time
import ctypes
import numpy as np
from typing import List

from picograd.print_utils import color_green
from .cuda import CudaDevice

def flatten_tensor(T: np.ndarray) -> np.ndarray:
  """Flatten tensor while preserving memory order."""
  return T.ravel()

def allocate_device_memory(manager: CudaDevice, T: np.ndarray) -> ctypes.c_void_p:
  """Allocate device memory for tensor."""
  return manager.cuda_malloc(T.nbytes)

def copy_data_to_device(manager: CudaDevice, d_T: ctypes.c_void_p, T_flat: np.ndarray):
  """Copy data from host to device."""
  manager.memcpy_htod(d_T, T_flat.ctypes.data, T_flat.nbytes)

# TODO: to be used for to(Device.CPU) from cuda
def free_device_tensor(manager: CudaDevice, d_T: ctypes.c_void_p):
  """Free tensor from device memory."""
  manager.cuda_free(d_T)

def tensor_to_cuda(tensor: "Tensor") -> "Tensor":
  start_time = time.time()
  T_flat = flatten_tensor(tensor.data)
  d_T = allocate_device_memory(tensor.device.manager, T_flat)
  copy_data_to_device(tensor.device.manager, d_T, T_flat)

  if tensor.device.manager.debug == 2:
    print(f"{color_green("[Cuda]")} Tensor data copied to device - {tensor.data.nbytes} bytes - {(time.time() - start_time) * 1000:.4f} ms")

  return d_T

def prep_kargs(
  d_A: ctypes.c_void_p,
  d_B: ctypes.c_void_p,
  d_C: ctypes.c_void_p,
  dim1: int,
  dim2: int,
  dim3: int
) -> List[ctypes.c_void_p]:
  """"Prepare kernel arguments."""
  return [
    ctypes.c_void_p(d_A.value),
    ctypes.c_void_p(d_B.value),
    ctypes.c_void_p(d_C.value),
    ctypes.c_int(dim1),
    ctypes.c_int(dim2),
    ctypes.c_int(dim3)
  ]
