import numpy as np

from .module import Module
from picograd import Tensor


class Dropout(Module):
  def __init__(self, p: float = 0.5):
    super().__init__()
    self.p = Tensor([p])

  def __call__(self, x):
    if self.train:
      mask = Tensor((np.random.rand(*x.shape) >= self.p.data).astype(x.dtype), device=x.device)
      return x * mask / (1.0 - self.p)
    else:
      return x
