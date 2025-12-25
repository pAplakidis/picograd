import os
import time
import ctypes
import numpy as np
from typing import Optional, Tuple, List
from enum import Enum, auto

from picograd.print_utils import *

DEBUG = int(os.getenv("DEBUG", 0))
PSEUDO_DEBUG = int(os.getenv("PSEUDO_DEBUG", 0))  # if 1, generate assembly code as string but don't print (helps with segfaults)

class Devices(Enum):
  CPU = auto()
  CUDA = auto()

  # CLANG = auto()
  # OPENCL = auto()
  # ROCM = auto()
  # XLA = auto()
  # TPU = auto()
  # SYCL = auto()
  # HIP = auto()
  # CUDA_JIT = auto()
  # ROCM_JIT = auto()
  # XLA_JIT = auto()

  def __str__(self): return self.name

class Device:
  def __init__(self, name: Devices, debug: int = DEBUG):
    self.name = name
    self.debug = debug

    if name == Devices.CPU:
      self.manager = None
    elif name == Devices.CUDA:
      from picograd.backend.cuda.cuda import CudaDeviceManager
      self.manager = CudaDeviceManager(name, debug=debug)
    else:
      raise NotImplementedError(f"Device {name} not implemented")

  def __str__(self): return str(self.name)

  def __repr__(self): return f"Device({self.name})"

class DeviceManager:
  def __init__(self, device_name: str, debug=DEBUG):
    self.device_name = device_name
    self.debug = debug

  @staticmethod
  def flatten_tensor(T: np.ndarray) -> np.ndarray:
    """Flatten tensor while preserving memory order."""
    return T.ravel()

  @staticmethod
  def prep_kargs(*args, **kwargs) -> List[ctypes.c_void_p]:
    """Prepare kernel arguments."""
    return [ctypes.c_void_p(arg.value) if isinstance(arg, ctypes.c_void_p) else ctypes.c_int(arg) for arg in args]

  @staticmethod
  def compute_gflops(num_elements: int, milliseconds: int):
    seconds = milliseconds / 1000.0
    return num_elements / (seconds * 1e9)

  def np_to_device(self, array: np.ndarray) -> Tuple[np.ndarray, ctypes.c_void_p]:
    """Copy a NumPy array to device memory."""
    array_flat = self.flatten_tensor(array)
    d_array = self.allocate_device_memory(array_flat)
    self.copy_data_to_device(d_array, array_flat)
    return array_flat, d_array

  def np_to_host(self, d_array: ctypes.c_void_p, array_flat: np.ndarray, shape: Optional[Tuple] = None) -> np.ndarray:
    """Copy a NumPy array from device memory to host."""
    self.copy_data_to_host(d_array, array_flat)
    self.free_device_tensor(d_array)
    return array_flat.reshape(shape) if shape else array_flat

  def tensor_to_device(self, tensor: "Tensor"):
    if tensor.device_data is None: self.host_data_to_dev(tensor)
    if tensor.requires_grad and tensor.device_grad is None: self.host_grad_to_dev(tensor)

  def host_data_to_dev(self, tensor: "Tensor"):
    assert tensor._data is not None, "Tensor data is None, cannot copy to device"

    if tensor.device.manager.debug >= 1 and not PSEUDO_DEBUG:
      start_time = time.time()

    T_flat = self.flatten_tensor(tensor._data)
    d_T = self.allocate_device_memory(T_flat)
    self.copy_data_to_device(d_T, T_flat)
    tensor.device_data = d_T
    
    if self.debug >= 1 and not PSEUDO_DEBUG:
      print(f"{color_green('[Cuda]')} Tensor data copied to device - {color_red(f'{tensor._data.nbytes} bytes')} - {color_red(f'{(time.time() - start_time) * 1000:.4f} ms')}")

  def host_grad_to_dev(self, tensor: "Tensor"):
    if tensor.device.manager.debug >= 1 and not PSEUDO_DEBUG:
      start_time = time.time()

    grad_flat = self.flatten_tensor(tensor.grad)
    d_grad = self.allocate_device_memory(grad_flat)
    self.copy_data_to_device(d_grad, grad_flat)
    tensor.device_grad = d_grad

    if self.debug >= 1 and not PSEUDO_DEBUG:
      print(f"{color_green('[Cuda]')} Tensor gradient copied to device - {color_red(f'{tensor._grad.nbytes} bytes')} - {color_red(f'{(time.time() - start_time) * 1000:.4f} ms')}")

  def dev_data_to_host(self, tensor: "Tensor", free=True):
    assert tensor.device_data is not None, "Tensor device data is None, cannot copy to host"

    if tensor.device.manager.debug >= 1 and not PSEUDO_DEBUG:
      start_time = time.time()

    data_flat = np.empty(tensor._shape, dtype=tensor.dtype).ravel()
    self.copy_data_to_host(tensor.device_data, data_flat)
    tensor._data = data_flat.reshape(tensor._shape)

    if free:
      self.free_device_tensor(tensor.device_data)
      tensor.device_data = None

    if tensor.device.manager.debug >= 1 and not PSEUDO_DEBUG:
      print(f"{color_green('[Cuda]')} Tensor data copied to host - {color_red(f'{tensor._data.nbytes} bytes')} - {color_red(f'{(time.time() - start_time) * 1000:.4f} ms')}")

  def dev_grad_to_host(self, tensor: "Tensor", free=True):
    assert tensor.device_grad is not None, "Tensor device grad is None, cannot copy to host"

    if tensor.device.manager.debug >= 1 and not PSEUDO_DEBUG:
      start_time = time.time()

    grad_flat = np.empty(tensor._shape, dtype=tensor.dtype).ravel()
    self.copy_data_to_host(tensor.device_grad, grad_flat)
    tensor._grad = grad_flat.reshape(tensor._shape)  # FIXME: will not work with CrossEntropyLoss (data.shape != grad.shape)

    if free:
      self.free_device_tensor(tensor.device_grad)
      tensor.device_grad = None

    if tensor.device.manager.debug >= 1 and not PSEUDO_DEBUG:
      print(f"{color_green('[Cuda]')} Tensor gradient copied to host - {color_red(f'{tensor._grad.nbytes} bytes')} - {color_red(f'{(time.time() - start_time) * 1000:.4f} ms')}")

  def tensor_to_host(self, tensor: "Tensor"):
    """Copy tensor data and gradient from device to host."""
    self.dev_data_to_host(tensor)
    self.dev_grad_to_host(tensor)

  # GENERIC DEVICE INTERFACE METHODS
  def allocate_device_memory(self, T: np.ndarray) -> ctypes.c_void_p: raise NotImplementedError("allocate_device_memory is not implemented for this device manager")
  def copy_data_to_device(self, d_T: ctypes.c_void_p, T_flat: np.ndarray): raise NotImplementedError("copy_data_to_device is not implemented for this device manager")
  def copy_data_to_host(self, d_T: ctypes.c_void_p, T_flat: np.ndarray): raise NotImplementedError("copy_data_to_host is not implemented for this device manager")
  def free_device_tensor(self, d_T: ctypes.c_void_p): raise NotImplementedError("free_device_tensor is not implemented for this device manager")
