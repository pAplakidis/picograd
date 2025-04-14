from enum import Enum, auto

class Device(Enum):
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
