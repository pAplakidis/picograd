import os
import time
import functools
import importlib
import numpy as np

from .device import Devices
from picograd.print_utils import *

DEBUG = int(os.getenv("DEBUG", 0))
PSEUDO_DEBUG = int(os.getenv("PSEUDO_DEBUG", 0))  # if 1, generate assembly code as string but don't print (helps with segfaults)

class Function:
  def __init__(self, device: Devices = Devices.CPU):
    self.device = device

    if device == Devices.CPU:
      module_path = "picograd.backend.cpu.ops"
    else:
      module_path = "picograd.backend.cuda.ops"
    ops_module = importlib.import_module(module_path)
    for name in dir(ops_module):
      if not name.startswith("_"): setattr(self, name, getattr(ops_module, name))

  @staticmethod
  def check_same_device(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
      devices = {arg.device.name for arg in args if type(arg).__name__ == "Tensor" and hasattr(arg, "device")}
      if len(devices) > 1: raise RuntimeError(f"Device mismatch: found devices {devices}")
      return method(self, *args, **kwargs)
    return wrapper

  @staticmethod
  def log_time(method):
    functools.wraps(method)
    def wrapper(self, *args, **kwargs):
      start_time = time.time()
      result = method(self, *args, **kwargs)
      end_time = time.time()
      if DEBUG >= 1 and not PSEUDO_DEBUG:
        print(f"{color_yellow(f"[Function-Perf]")} {self.__class__.__name__}.{method.__name__} - {color_yellow(f"{(end_time - start_time) * 1000.0:.4f}")} ms")
      return result
    return wrapper

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise NotImplementedError(f"backward not implemented for {type(self)}")

  def __init_subclass__(cls):
    if 'forward' in cls.__dict__:
      cls.forward = Function.check_same_device(cls.forward)
    if 'backward' in cls.__dict__:
      cls.backward = Function.check_same_device(cls.backward)

    for method_name in ['forward', 'backward']:
      if method_name in cls.__dict__:
        method = getattr(cls, method_name)
        # Wrap first with check_same_device, then with log_time
        method = Function.check_same_device(method)
        method = cls.log_time(method)
        setattr(cls, method_name, method)

# BINARY OPS

class Add(Function):
  def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
    self.a, self.b = a, b
    return self.BinaryOps.add(a, b)

  def backward(self, grad_out: np.ndarray):
    self.BinaryOps().add_back(self.a, self.b, grad_out)

class Mul(Function):
  def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
    self.a, self.b = a, b
    return self.BinaryOps.mul(a, b)

  def backward(self, grad_out: np.ndarray):
    self.BinaryOps().mul_back(self.a, self.b, grad_out)

class Dot(Function):
  def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
    self.a, self.b = a, b
    return self.BinaryOps.dot(a, b)

  def backward(self, grad_out: np.ndarray):
    self.BinaryOps().dot_back(self.a, self.b, grad_out)

class Conv2D(Function):
  def forward(self, a: "Tensor", w: "Tensor", b: "Tensor",
              in_channels: int, out_channels: int, stride: int = 1, padding: int = 0) -> "Tensor":
    self.a, self.w, self.b = a, w, b
    self.stride, self.padding = stride, padding
    return self.BinaryOps.conv2d(a, w, b, in_channels, out_channels, stride, padding)
  
  def backward(self, grad_out: np.ndarray):
    self.BinaryOps.conv2d_back(self.a, grad_out, self.w, self.b, self.a.shape[1], self.w.shape[0], self.stride, self.padding)

# UNARY OPS

class ReLU(Function):
  def forward(self, a: "Tensor"):
    self.a = a
    return self.UnaryOps.relu(a)

  def backward(self, grad_out: np.ndarray):
    self.UnaryOps.relu_back(self.a, grad_out)

class Softmax(Function):
  def forward(self, a: "Tensor"):
    self.a = a
    self.out = self.UnaryOps.softmax(a)
    return self.out
  
  def backward(self, grad_out: np.ndarray):
    return self.UnaryOps.softmax_back(self.a, self.out, grad_out)

# REDUCE OPS

class Sum(Function):
  pass

class Mean(Function):
  pass

class Max(Function):
  pass

class Min(Function):
  pass

class MaxPool2D(Function):
  def forward(self, a: "Tensor", filter=(2, 2), stride=1):
    self.a = a
    self.filter = filter
    self.stride = stride
    ret, self.mask = self.ReduceOps.maxpool2d(a, filter, stride)
    return ret

  def backward(self, grad_out: np.ndarray):
    self.ReduceOps.maxpool2d_back(self.a, grad_out, self.mask, self.filter, self.stride)

class AvgPool2D(Function):
  def forward(self, a: "Tensor", filter=(2, 2), stride=1):
    self.a = a
    self.filter = filter
    self.stride = stride
    return self.ReduceOps.avgpool2d(a, filter, stride)

  def backward(self, grad_out: np.ndarray):
    self.ReduceOps.avgpool2d_back(self.a, grad_out, self.filter, self.stride)

class CrossEntropy(Function):
  def forward(self, z: "Tensor", y: "Tensor") -> "Tensor":
    self.z = z
    self.y = y
    return self.ReduceOps.cross_entropy(z, y)
  
  def backward(self):
    self.ReduceOps.cross_entropy_back(self.z, self.y)
