import picograd.nn as nn
from picograd import Tensor

class ReLU(nn.Module):
  def __init__(self):
    super().__init__()

  def __call__(self, x: Tensor) -> Tensor:
    return x.relu()
