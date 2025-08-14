import os
import time
import ctypes
import numpy as np
from typing import List, Tuple, Optional

from picograd.print_utils import *
from .cuda import CudaDevice

PSEUDO_DEBUG = int(os.getenv("PSEUDO_DEBUG", 0))  # if 1, generate assembly code as string but don't print (helps with segfaults)
TILE_SIZE = 16

# TODO: expose as generic device API to avoid if statements in tensor

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

def free_device_tensor(manager: CudaDevice, d_T: ctypes.c_void_p):
  """Free tensor from device memory."""
  manager.cuda_free(d_T)

def np_to_device(array: np.ndarray, manager: CudaDevice) -> Tuple[np.ndarray, ctypes.c_void_p]:
  """Copy a NumPy array to device memory."""
  array_flat = flatten_tensor(array)
  d_array = allocate_device_memory(manager, array_flat)
  copy_data_to_device(manager, d_array, array_flat)
  return array_flat, d_array

def np_to_host(d_array: ctypes.c_void_p, array_flat: np.ndarray, manager: CudaDevice, shape: Optional[Tuple] = None) -> np.ndarray:
  """Copy a NumPy array from device memory to host."""
  manager.memcpy_dtoh(array_flat.ctypes.data, d_array, array_flat.nbytes)
  free_device_tensor(manager, d_array)
  return array_flat.reshape(shape) if shape else array_flat

def tensor_to_device(tensor: "Tensor"):
  assert tensor.data is not None, "Tensor data is None, cannot copy to device"

  if tensor.device.manager.debug >= 1 and not PSEUDO_DEBUG:
    start_time = time.time()

  T_flat = flatten_tensor(tensor.data)
  d_T = allocate_device_memory(tensor.device.manager, T_flat)
  copy_data_to_device(tensor.device.manager, d_T, T_flat)
  tensor.device_data = d_T

  grad_flat = flatten_tensor(tensor.grad)
  d_grad = allocate_device_memory(tensor.device.manager, grad_flat)
  copy_data_to_device(tensor.device.manager, d_grad, grad_flat)
  tensor.device_grad = d_grad

  if tensor.device.manager.debug >= 1 and not PSEUDO_DEBUG:
    print(f"{color_green("[Cuda]")} Tensor data and gradient copied to device - {color_red(f"{tensor._data.nbytes} bytes, {tensor._grad.nbytes} bytes")} - {color_red(f"{(time.time() - start_time) * 1000:.4f} ms")}")

def dev_data_to_host(tensor: "Tensor", free=True):
  assert tensor.device_data is not None, "Tensor device data is None, cannot copy to host"

  if tensor.device.manager.debug >= 1 and not PSEUDO_DEBUG:
    start_time = time.time()

  if tensor._data is None: 
    tensor._data = np.empty(tensor._shape, dtype=tensor.dtype).ravel()

  copy_data_to_host(tensor.device.manager, tensor.device_data, tensor._data)
  if free:
    free_device_tensor(tensor.device.manager, tensor.device_data)
    tensor.device_data = None

  tensor._data = tensor._data.reshape(tensor._shape)

  if tensor.device.manager.debug >= 1 and not PSEUDO_DEBUG:
    print(f"{color_green("[Cuda]")} Tensor data copied to host - {color_red(f"{tensor._data.nbytes} bytes, {tensor._grad.nbytes} bytes")} - {color_red(f"{(time.time() - start_time) * 1000:.4f} ms")}")

def dev_grad_to_host(tensor: "Tensor", free=True):
  assert tensor.device_grad is not None, "Tensor device grad is None, cannot copy to host"

  if tensor.device.manager.debug >= 1 and not PSEUDO_DEBUG:
    start_time = time.time()

  if tensor._data is None: 
    tensor._data = np.empty(tensor._shape, dtype=tensor.dtype).ravel()

  # FIXME: grad flat
  copy_data_to_host(tensor.device.manager, tensor.device_grad, tensor._grad)
  if free:
    free_device_tensor(tensor.device.manager, tensor.device_grad)
    tensor.device_grad = None
  tensor._grad.reshape(tensor._shape)  # FIXME: will not work with CrossEntropyLoss (data.shape != grad.shape)

  if tensor.device.manager.debug >= 1 and not PSEUDO_DEBUG:
    print(f"{color_green("[Cuda]")} Tensor gradient copied to host - {color_red(f"{tensor._data.nbytes} bytes, {tensor._grad.nbytes} bytes")} - {color_red(f"{(time.time() - start_time) * 1000:.4f} ms")}")

def tensor_to_host(tensor: "Tensor"):
  """Copy tensor data and gradient from device to host."""
  dev_data_to_host(tensor)
  dev_grad_to_host(tensor)

@staticmethod
def prep_kargs(*args, **kwargs) -> List[ctypes.c_void_p]:
  """Prepare kernel arguments."""
  return [ctypes.c_void_p(arg.value) if isinstance(arg, ctypes.c_void_p) else ctypes.c_int(arg) for arg in args]

def compute_gflops(num_elements: int, milliseconds: int):
  seconds = milliseconds / 1000.0
  return num_elements / (seconds * 1e9)
