import os
import time
import functools
import importlib
from enum import Enum, auto

from .device import Devices
from picograd.print_utils import *

DEBUG = int(os.getenv("DEBUG", 0))
PSEUDO_DEBUG = int(os.getenv("PSEUDO_DEBUG", 0))  # if 1, generate assembly code as string but don't print (helps with segfaults)


class OPS(Enum):
  # Binary ops
  ADD = auto()
  MUL = auto()
  DOT = auto()
  POW = auto()
  Conv2D = auto()

  # Unary ops
  ReLU = auto()
  Tanh = auto()
  Softmax = auto()
  Sigmoid = auto()

  # Substitution ops
  Reshape = auto()
  Flatten = auto()
  Unsqueeze = auto()
  Squeeze = auto()
  
  # Reduce ops
  SUM = auto()
  MEAN = auto()
  MAX = auto()
  MIN = auto()
  STD = auto()
  ARGMAX = auto()
  ARGMIN = auto()

  MaxPool2D = auto()
  AvgPool2D = auto()

  MSELoss = auto()
  MAELoss = auto()
  CrossEntropyLoss = auto()
  BCELoss = auto()

  def __str__(self): return self.name


def get_op(op_name: str, device_name: str):
  # Binary Ops
  if op_name == OPS.ADD: return Add(device_name)
  if op_name == OPS.MUL: return Mul(device_name)
  if op_name == OPS.DOT: return Dot(device_name)
  if op_name == OPS.Conv2D: return Conv2D(device_name)

  # Unary Ops
  if op_name == OPS.ReLU: return ReLU(device_name)
  if op_name == OPS.Softmax: return Softmax(device_name)

  # Reduce Ops
  if op_name == OPS.SUM: return Sum(device_name)
  if op_name == OPS.MEAN: return Mean(device_name)
  if op_name == OPS.MAX: return Max(device_name)
  if op_name == OPS.MIN: return Min(device_name)
  if op_name == OPS.ARGMAX: return Argmax(device_name)
  if op_name == OPS.ARGMIN: return Argmin(device_name)
  if op_name == OPS.STD: return Std(device_name)
  if op_name == OPS.MaxPool2D: return MaxPool2D(device_name)
  if op_name == OPS.AvgPool2D: return AvgPool2D(device_name)
  if op_name == OPS.CrossEntropyLoss: return CrossEntropy(device_name)

  raise ValueError(f"Unknown op {op_name}")


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
        print(f"{color_yellow('[Function-Perf]')} {self.__class__.__name__}.{method.__name__} - {color_yellow(f'{(end_time - start_time) * 1000.0:.4f}')} ms")
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

  def backward(self, grad_out):
    self.BinaryOps().add_back(self.a, self.b, grad_out)

class Mul(Function):
  def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
    self.a, self.b = a, b
    return self.BinaryOps.mul(a, b)

  def backward(self, grad_out):
    self.BinaryOps().mul_back(self.a, self.b, grad_out)

class Dot(Function):
  def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
    self.a, self.b = a, b
    return self.BinaryOps.dot(a, b)

  def backward(self, grad_out):
    self.BinaryOps().dot_back(self.a, self.b, grad_out)

class Conv2D(Function):
  def forward(self, a: "Tensor", w: "Tensor", b: "Tensor",
              in_channels: int, out_channels: int, stride: int = 1, padding: int = 0) -> "Tensor":
    self.a, self.w, self.b = a, w, b
    self.stride, self.padding = stride, padding
    return self.BinaryOps.conv2d(a, w, b, in_channels, out_channels, stride, padding)
  
  def backward(self, grad_out):
    self.BinaryOps.conv2d_back(self.a, grad_out, self.w, self.b, self.a.shape[1], self.w.shape[0], self.stride, self.padding)

# UNARY OPS

class ReLU(Function):
  def forward(self, a: "Tensor"):
    self.a = a
    return self.UnaryOps.relu(a)

  def backward(self, grad_out):
    self.UnaryOps.relu_back(self.a, grad_out)

class Softmax(Function):
  def forward(self, a: "Tensor"):
    self.a = a
    self.out = self.UnaryOps.softmax(a)
    return self.out
  
  def backward(self, grad_out):
    return self.UnaryOps.softmax_back(self.a, self.out, grad_out)

class BatchNorm2D(Function):
  def forward(self, a: "Tensor", beta: "Tensor", gamma: "Tensor"):
    return self.UnaryOps.batchnorm2d(a)

  def backward(self, grad_out):
    pass

# REDUCE OPS

class Sum(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    return self.ReduceOps.sum(a, axis, keepdims)

  def backward(self, grad_out):
    pass

class Mean(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    return self.ReduceOps.mean(a, axis, keepdims)

  def backward(self, grad_out):
    pass
class Max(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    return self.ReduceOps.max(a, axis, keepdims)

  def backward(self, grad_out):
    pass
class Min(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    return self.ReduceOps.min(a, axis, keepdims)

  def backward(self, grad_out):
    pass

class Std(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    return self.ReduceOps.std(a, axis, keepdims)

  def backward(self, grad_out):
    pass

class Argmax(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    return self.ReduceOps.argmax(a, axis, keepdims)

  def backward(self, grad_out):
    pass

class Argmin(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    return self.ReduceOps.argmin(a, axis, keepdims)

  def backward(self, grad_out):
    pass

class MaxPool2D(Function):
  def forward(self, a: "Tensor", filter=(2, 2), stride=1):
    self.a = a
    self.filter = filter
    self.stride = stride
    ret, self.mask = self.ReduceOps.maxpool2d(a, filter, stride)
    return ret

  def backward(self, grad_out):
    self.ReduceOps.maxpool2d_back(self.a, grad_out, self.mask, self.filter, self.stride)

class AvgPool2D(Function):
  def forward(self, a: "Tensor", filter=(2, 2), stride=1):
    self.a = a
    self.filter = filter
    self.stride = stride
    return self.ReduceOps.avgpool2d(a, filter, stride)

  def backward(self, grad_out):
    self.ReduceOps.avgpool2d_back(self.a, grad_out, self.filter, self.stride)

class CrossEntropy(Function):
  def forward(self, z: "Tensor", y: "Tensor") -> "Tensor":
    self.z = z
    self.y = y
    return self.ReduceOps.cross_entropy(z, y)
  
  def backward(self):
    self.ReduceOps.cross_entropy_back(self.z, self.y)
