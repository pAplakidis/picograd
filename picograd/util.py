import subprocess
from typing import Tuple

MAX_GRAD_NORM = 1.0

def get_key_from_value(d, val): return [k for k, v in d.items() if v == val]

def is_cuda_available() -> bool:
  try:
    output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
    return True
  except (subprocess.CalledProcessError, FileNotFoundError):
    return False

def default_strides(shape: Tuple) -> Tuple:
  """ Get numpy-style contiguous row-major strides from shape (for float32 = 4 bytes) """
  stride = 1
  strides = []
  for dim in reversed(shape):
    strides.append(stride)
    stride *= dim
  return tuple(reversed(strides))

def get_real_shape(shape: Tuple, strides: Tuple) -> Tuple[int]:
  """ Get the actual shape of a tensor given its shape and strides, accounting for broadcasted dimensions """
  return tuple(dim for dim, stride in zip(shape, strides) if stride != 0)

def required_numel(shape: Tuple, strides: Tuple):
  """ Compute how much backing memory is required for a tensor with given shape and strides """
  max_offset = 0
  for dim, stride in zip(shape, strides):
      if dim == 0:
          return 0
      max_offset += (dim - 1) * stride
  return max_offset + 1

def broadcast_shape(a: Tuple[int, ...], b: Tuple[int, ...]):
  ndim = max(len(a), len(b))
  a = (1,) * (ndim - len(a)) + a
  b = (1,) * (ndim - len(b)) + b
  out = []
  for x, y in zip(a, b):
    assert x == y or x == 1 or y == 1, f"Cannot broadcast {a} and {b}"
    out.append(max(x, y))
  return tuple(out)
