import numpy as np
from typing import Tuple, Optional

from picograd.backend.ops import *

class Function:
  def __init__(self):
    self.device = None

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise NotImplementedError(f"backward not implemented for {type(self)}")

# TODO: np.ndarray => Tensor to check requires_grad
class Add(Function):
  def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    self.a, self.b = a, b
    return BinaryOps.add(a, b)

  def backward(self, grad_out: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]: return (grad_out, grad_out)

class Mul(Function):
  def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    self.a, self.b = a, b
    return BinaryOps.mul(a, b)

  def backward(self, grad_out: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]: return (self.a * grad_out, self.b * grad_out)

class Dot(Function):
  def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    self.a, self.b = a, b
    return np.dot(a, b)

  # TODO: this is different (?)
  def backward(self, grad_out: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]: return (self.a * grad_out, self.b * grad_out)

