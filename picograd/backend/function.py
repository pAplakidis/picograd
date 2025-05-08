import os
import time
import importlib
import numpy as np

from .device import Devices
from picograd.print_utils import *

DEBUG = int(os.getenv("DEBUG", 0))

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
    def wrapper(self, *args, **kwargs):
      devices = {arg.device for arg in args if hasattr(arg, 'device')}
      if len(devices) > 1: raise RuntimeError(f"Device mismatch: found devices {devices}")
      return method(self, *args, **kwargs)
    return wrapper

  @staticmethod
  def log_time(method):
    def wrapper(self, *args, **kwargs):
      start_time = time.time()
      result = method(self, *args, **kwargs)
      end_time = time.time()
      if DEBUG >= 2:
        print(f"{color_yellow(f"[Function-Perf]")} {self.__class__.__name__}.{method.__name__} - {(end_time - start_time) * 1000.0:.4f} ms")
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

class Add(Function):
  def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
    self.a, self.b = a, b
    return self.BinaryOps.add(a, b)

  def backward(self, grad_out: "Tensor"):
    self.BinaryOps().add_back(self.a, self.b, grad_out)

class Mul(Function):
  def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
    self.a, self.b = a, b
    return self.BinaryOps.mul(a, b)

  def backward(self, grad_out: "Tensor"):
    self.BinaryOps().mul_back(self.a, self.b, grad_out)

class Dot(Function):
  def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
    self.a, self.b = a, b
    return self.BinaryOps.dot(a, b)

  def backward(self, grad_out: "Tensor"):
    self.BinaryOps().dot_back(self.a, self.b, grad_out)

class Conv2D(Function):
  def forward(self, a: "Tensor", w: "Tensor", b: "Tensor",
              in_channels: int, out_channels: int, stride: int = 1, padding: int = 0) -> "Tensor":
    self.a, self.w, self.b = a, w, b
    self.stride, self.padding = stride, padding
    return self.BinaryOps.conv2d(a, w, b, in_channels, out_channels, stride, padding)
  
  def backward(self, grad_out: "Tensor"):
    grad_a, grad_w, grad_b = self.BinaryOps.conv2d_backward(self.a.data, grad_out, self.w.data, self.b.data, self.a.shape[1], self.w.shape[0], self.stride, self.padding)
    if self.a.requires_grad: self.a.grad = grad_a
    if self.w.requires_grad: self.w.grad = grad_w
    if self.b.requires_grad: self.b.grad = grad_b
