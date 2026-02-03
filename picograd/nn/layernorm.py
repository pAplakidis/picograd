from typing import Tuple
from .module import Layer, LayerType
from picograd.tensor import Tensor


class LayerNorm(Layer):
  def __init__(self, normalized_shape: Tuple[int], eps=1e-5):
    super().__init__()
    self.type = LayerType.LAYERNORM
    if isinstance(normalized_shape, int): normalized_shape = (normalized_shape,)
    self.normalized_shape = tuple(normalized_shape)
    self.eps = Tensor([eps], requires_grad=False, name="layernorm-eps", device=self.device)

    self.weight = Tensor.ones(self.normalized_shape, name="layernorm-gamma", device=self.device)
    self.bias   = Tensor.zeros(self.normalized_shape, name="layernorm-beta", device=self.device)

    self.register_param("weight", self.weight)
    self.register_param("bias", self.bias)

  def __call__(self, x: Tensor) -> Tensor:
    axes = tuple(range(-len(self.normalized_shape), 0)) # normalize across last len(normalized_shape) dims
    mean = x.mean(axis=axes, keepdims=True)
    std  = x.std(axis=axes, keepdims=True)
    x_norm = (x - mean) / (std+ self.eps).sqrt()
    x_norm.name = "x_norm"
    self.out = self.weight * x_norm + self.bias
    return self.out


