import numpy as np
from .module import Layer, LayerType
from picograd.tensor import Tensor


class Linear(Layer):
  def __init__(self, in_feats: int, out_feats: int, bias: bool = True, initialization: str = 'gaussian'):
    super().__init__()
    self.type = LayerType.LINEAR
    self.in_feats = in_feats
    self.out_feats = out_feats

    if initialization == "gaussian":
      self.weight = Tensor(0.01 * np.random.randn(self.in_feats, self.out_feats), name="linear-weight", device=self.device)
    elif initialization == "xavier":
      self.weight = Tensor(np.random.randn(self.in_feats, self.out_feats) * np.sqrt(2. / (self.in_feats + self.out_feats)), name="linear-weight", device=self.device)
    else:
      raise ValueError("Invalid initialization method")
    
    self.bias = Tensor(np.zeros((self.out_feats,)), name="linear-bias", device=self.device) if bias else None

    self.register_param("weight", self.weight)
    if bias:
      self.register_param("bias", self.bias)

  def __call__(self, x: Tensor) -> Tensor:
    assert len(x.shape) >= 2, "Input Tensor requires batch_size dimension"
    self.t_in = x
    self.t_out = x.linear(self.weight, self.bias)
    return self.t_out
    

