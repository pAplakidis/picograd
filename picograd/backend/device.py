import os
from enum import Enum, auto

DEBUG = int(os.getenv("DEBUG", 0))

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
      from picograd.backend.cuda.cuda import CudaDevice
      self.manager = CudaDevice(debug=debug)
    else:
      raise NotImplementedError(f"Device {name} not implemented")

  def __str__(self): return str(self.name)

  def __repr__(self): return f"Device({self.name})"

