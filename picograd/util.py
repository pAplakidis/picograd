import subprocess

MAX_GRAD_NORM = 1.0

def get_key_from_value(d, val): return [k for k, v in d.items() if v == val]

def is_cuda_available():
  try:
    output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
    return True
  except (subprocess.CalledProcessError, FileNotFoundError):
    return False
