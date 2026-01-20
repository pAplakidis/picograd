from .module import Layer, LayerType
from picograd.tensor import Tensor


class MaxPool2D(Layer):
  def __init__(self, filter=(2,2), stride=1):
    super().__init__()
    self.type = LayerType.MAXPOOL2D
    
    self.filter = filter
    self.stride = stride

  def __call__(self, x: Tensor) -> Tensor:
    self.t_in = x
    self.t_out = x.maxpool2d(self.filter, self.stride)
    return self.t_out


class AvgPool2D(Layer):
  def __init__(self, filter=(2,2), stride=1, padding=0):
    super().__init__()
    self.type = LayerType.AVGPOOL2D
    
    self.filter = filter
    self.stride = stride
    self.padding = padding

  def __call__(self, x: Tensor) -> Tensor:
    self.t_in = x
    self.t_out = x.avgpool2d(self.filter, self.stride, self.padding)
    return self.t_out

