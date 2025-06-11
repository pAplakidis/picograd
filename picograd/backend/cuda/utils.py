import time
import ctypes
import numpy as np
from typing import List

from picograd.print_utils import *
from .cuda import CudaDevice

TILE_SIZE = 16

def flatten_tensor(T: np.ndarray) -> np.ndarray:
  """Flatten tensor while preserving memory order."""
  return T.ravel()

def allocate_device_memory(manager: CudaDevice, T: np.ndarray) -> ctypes.c_void_p:
  """Allocate device memory for tensor."""
  return manager.cuda_malloc(T.nbytes)

def copy_data_to_device(manager: CudaDevice, d_T: ctypes.c_void_p, T_flat: np.ndarray):
  """Copy data from host to device."""
  manager.memcpy_htod(d_T, T_flat.ctypes.data, T_flat.nbytes)

def copy_data_to_host(manager: CudaDevice, d_T: ctypes.c_void_p, T_flat: np.ndarray):
  """Copy data from device to host."""
  manager.memcpy_dtoh(T_flat.ctypes.data, d_T, T_flat.nbytes)

# TODO: to be used for to(Device.CPU) from cuda
def free_device_tensor(manager: CudaDevice, d_T: ctypes.c_void_p):
  """Free tensor from device memory."""
  manager.cuda_free(d_T)

def tensor_to_cuda(tensor: "Tensor") -> "Tensor":
  start_time = time.time()
  T_flat = flatten_tensor(tensor.data)
  d_T = allocate_device_memory(tensor.device.manager, T_flat)
  copy_data_to_device(tensor.device.manager, d_T, T_flat)

  grad_flat = flatten_tensor(tensor.grad)
  d_grad = allocate_device_memory(tensor.device.manager, grad_flat)
  copy_data_to_device(tensor.device.manager, d_grad, grad_flat)

  if tensor.device.manager.debug >= 1:
    print(f"{color_green("[Cuda]")} Tensor data and gradient copied to device - {color_red(f"{tensor.data.nbytes} bytes, {tensor.grad.nbytes} bytes")} - {color_red(f"{(time.time() - start_time) * 1000:.4f} ms")}")

  return d_T, d_grad

@staticmethod
def prep_kargs(*args, **kwargs) -> List[ctypes.c_void_p]:
  """Prepare kernel arguments."""
  return [ctypes.c_void_p(arg.value) if isinstance(arg, ctypes.c_void_p) else ctypes.c_int(arg) for arg in args]

def compute_gflops(num_elements: int, milliseconds: int):
  seconds = milliseconds / 1000.0
  return num_elements / (seconds * 1e9)
