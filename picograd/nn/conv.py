import numpy as np
from .module import Layer, LayerType
from picograd.tensor import Tensor


  # TODO: write this
class Conv1D(Layer):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0):
    super.__init__()
    self.type = LayerType.CONV1D
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

  def __call__(self, x: Tensor) -> Tensor:
    return None


class Conv2D(Layer):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0):
    super().__init__()
    self.type = LayerType.CONV2D
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    assert self.kernel_size % 2 != 0, "Conv2D kenrel_size must be odd"
    assert self.kernel_size in [3, 5, 7, 9]

    self.weight = Tensor(np.random.uniform(0.0, 1.0, (out_channels, in_channels, kernel_size, kernel_size)), "conv2D_kernel", device=self.device)
    self.bias = Tensor(np.zeros((out_channels,)), name="bias", device=self.device)

    self.register_param("weight", self.weight)
    self.register_param("bias", self.bias)

  def __call__(self, x: Tensor) -> Tensor:
    self.t_in = x
    self.t_out = x.conv2d(self.weight, self.in_channels, self.out_channels, self.stride, self.padding, bias=self.bias)
    return self.t_out


