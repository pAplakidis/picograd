import subprocess

MAX_GRAD_NORM = 1.0

def get_key_from_value(d, val): return [k for k, v in d.items() if v == val]

def is_cuda_available() -> bool:
  try:
    output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
    return True
  except (subprocess.CalledProcessError, FileNotFoundError):
    return False

def default_strides(shape: tuple) -> tuple:
  """ Get numpy-style contiguous row-major strides from shape (for float32 = 4 bytes) """
  stride = 1
  strides = []
  for dim in reversed(shape):
    strides.append(stride)
    stride *= dim
  return tuple(reversed(strides))

def check_contiguous(shape: tuple, strides: tuple) -> bool:
  return strides == default_strides(shape)
