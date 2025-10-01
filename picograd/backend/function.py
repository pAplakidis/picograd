import os
import time
import functools
import importlib
from typing import Tuple
from enum import Enum, auto

from .device import Devices
from picograd.print_utils import *

DEBUG = int(os.getenv("DEBUG", 0))
PSEUDO_DEBUG = int(os.getenv("PSEUDO_DEBUG", 0))  # if 1, generate assembly code as string but don't print (helps with segfaults)


# TODO: split into BOps, UOps, MOps, ROps classes
class OPS(Enum):
  # Binary ops
  ADD = auto()
  MUL = auto()
  DOT = auto()
  POW = auto()
  Conv2D = auto()

  # Unary ops
  ReLU = auto()
  Softmax = auto()
  Tanh = auto()
  Sigmoid = auto()

  # Movement ops
  Reshape = auto()
  View = auto()
  Flatten = auto()
  Unsqueeze = auto()
  Squeeze = auto()
  Transpose = auto()
  Cat = auto()
  
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
  if op_name == OPS.ADD:    return Add(device_name)
  if op_name == OPS.MUL:    return Mul(device_name)
  if op_name == OPS.DOT:    return Dot(device_name)
  if op_name == OPS.POW:    return Pow(device_name)
  if op_name == OPS.Conv2D: return Conv2D(device_name)

  # Unary Ops
  if op_name == OPS.ReLU:    return ReLU(device_name)
  if op_name == OPS.Softmax: return Softmax(device_name)
  if op_name == OPS.Tanh:    return Tanh(device_name)
  if op_name == OPS.Sigmoid: return Sigmoid(device_name)

  # Reduce Ops
  if op_name == OPS.SUM:       return Sum(device_name)
  if op_name == OPS.MEAN:      return Mean(device_name)
  if op_name == OPS.MAX:       return Max(device_name)
  if op_name == OPS.MIN:       return Min(device_name)
  if op_name == OPS.ARGMAX:    return Argmax(device_name)
  if op_name == OPS.ARGMIN:    return Argmin(device_name)
  if op_name == OPS.STD:       return Std(device_name)
  if op_name == OPS.MaxPool2D: return MaxPool2D(device_name)
  if op_name == OPS.AvgPool2D: return AvgPool2D(device_name)
  if op_name == OPS.CrossEntropyLoss: return CrossEntropy(device_name)

  # Movement Ops
  if op_name == OPS.Reshape:   return Reshape(device_name)
  if op_name == OPS.View:      return View(device_name)
  if op_name == OPS.Flatten:   return Flatten(device_name)
  if op_name == OPS.Unsqueeze: return Unsqueeze(device_name)
  if op_name == OPS.Squeeze:   return Squeeze(device_name)
  if op_name == OPS.Transpose: return Transpose(device_name)

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

class Pow(Function):
  def forward(self, a: "Tensor", b: float) -> "Tensor":
    self.a, self.b = a, b
    return self.BinaryOps.pow(a, b)

  def backward(self, grad_out):
    self.BinaryOps().pow_back(self.a, self.b, grad_out)

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
  def forward(self, a: "Tensor", axis=None):
    self.a = a
    self.axis = axis if axis is not None else -1
    self.out = self.UnaryOps.softmax(a, axis)
    return self.out
  
  def backward(self, grad_out):
    return self.UnaryOps.softmax_back(self.a, self.out, grad_out, self.axis)

class Tanh(Function):
  def forward(self, a: "Tensor"):
    self.a = a
    return self.UnaryOps.tanh(a)

  def backward(self, grad_out):
    self.UnaryOps.tanh_back(self.a, grad_out)

class Sigmoid(Function):
  def forward(self, a: "Tensor"):
    self.a = a
    self.out = self.UnaryOps.sigmoid(a)
    return self.out

  def backward(self, grad_out):
    self.UnaryOps.sigmoid_back(self.a, self.out, grad_out)


# REDUCE OPS
class Sum(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    self.a = a
    self.axis = axis
    self.keepdims = keepdims
    return self.ReduceOps.sum(a, axis, keepdims)

  def backward(self, grad_out):
    self.ReduceOps.sum_back(self.a, grad_out, self.axis, self.keepdims)

class Mean(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    self.a = a
    self.axis = axis
    self.keepdims = keepdims
    return self.ReduceOps.mean(a, axis, keepdims)

  def backward(self, grad_out):
    self.ReduceOps.mean_back(self.a, grad_out, self.axis, self.keepdims)

class Std(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    self.a = a
    self.axis = axis
    self.keepdims = keepdims
    return self.ReduceOps.std(a, axis, keepdims)

  def backward(self, grad_out):
    self.ReduceOps.std_back(self.a, grad_out, self.axis, self.keepdims)

class Max(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    self.a = a
    self.axis = axis
    self.keepdims = keepdims
    return self.ReduceOps.max(a, axis, keepdims)

  def backward(self, grad_out):
    self.ReduceOps.max_back(self.a, grad_out, self.axis, self.keepdims)

class Min(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    self.a = a
    self.axis = axis
    self.keepdims = keepdims
    return self.ReduceOps.min(a, axis, keepdims)

  def backward(self, grad_out):
    self.ReduceOps.min_back(self.a, grad_out, self.axis, self.keepdims)

class Argmax(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    self.a = a
    return self.ReduceOps.argmax(a, axis, keepdims)

  def backward(self, grad_out):
    self.ReduceOps.argmax_back(self.a)

class Argmin(Function):
  def forward(self, a: "Tensor", axis=None, keepdims=False):
    self.a = a
    return self.ReduceOps.argmin(a, axis, keepdims)

  def backward(self, grad_out):
    self.ReduceOps.argmin_back(self.a)

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


# MOVEMENT OPS
class Reshape(Function):
  def forward(self, a: "Tensor", shape):
    self.a = a
    self.original_shape = a.shape
    return self.MovementOps.reshape(a, shape)
  
  def backward(self, grad_out):
    self.MovementOps.reshape_back(self.a, grad_out, self.original_shape)

class View(Function):
  def forward(self, a: "Tensor", shape):
    self.a = a
    self.original_shape = a.shape
    return self.MovementOps.view(a, shape)
  
  def backward(self, grad_out):
    self.MovementOps.view_back(self.a, grad_out, self.original_shape)

class Flatten(Function):
  def forward(self, a: "Tensor"):
    self.a = a
    self.original_shape = a.shape
    return self.MovementOps.flatten(a)
  
  def backward(self, grad_out):
    self.MovementOps.flatten_back(self.a, grad_out, self.original_shape)

class Unsqueeze(Function):
  def forward(self, a: "Tensor", axis):
    self.a = a
    self.axis = axis
    self.original_shape = a.shape
    return self.MovementOps.unsqueeze(a, axis)
  
  def backward(self, grad_out):
    self.MovementOps.unsqueeze_back(self.a, grad_out, self.axis, self.original_shape)

class Squeeze(Function):
  def forward(self, a: "Tensor", axis):
    self.a = a
    self.axis = axis
    self.original_shape = a.shape
    return self.MovementOps.squeeze(a, axis)
  
  def backward(self, grad_out):
    self.MovementOps.squeeze_back(self.a, grad_out, self.axis, self.original_shape)

class Transpose(Function):
  def forward(self, a: "Tensor", axes: Tuple[int] = None):
    self.a = a
    self.axes = axes
    return self.MovementOps.transpose(a, axes)
  
  def backward(self, grad_out):
    self.MovementOps.transpose_back(self.a, grad_out, self.axes)
