from .module import Layer, LayerType
from picograd.tensor import Tensor

class BatchNorm1D(Layer):
  def __init__(self, n_feats: int, eps=1e-5, momentum=0.1):
    super().__init__()
    self.type = LayerType.BATCHNORM2D
    self.subgraph_name = "BatchNorm1D"
    self.n_feats = n_feats
    self.eps      = Tensor([eps], requires_grad=False, name="batchnorm1d-eps", device=self.device)
    self.momentum = Tensor([momentum], requires_grad=False, name="batchnorm1d-momentum", device=self.device)

    self.weight = Tensor.ones((1, self.n_feats), name="batchnorm2d-gamma", device=self.device)
    self.bias   = Tensor.zeros((1, self.n_feats), name="batchnorm2d-beta", device=self.device)

    self.running_mean = Tensor.zeros((1, self.n_feats), requires_grad=False, name="batchnorm1d-running-mean", device=self.device)
    self.running_var  = Tensor.ones((1, self.n_feats), requires_grad=False, name="batchnorm1d-running-var", device=self.device)

    self.register_param("weight", self.weight)
    self.register_param("bias", self.bias)

  def __call__(self, x: Tensor) -> Tensor:
    assert len(x.shape) == 2, "BatchNorm1D requires input of shape (batch_size, features)"
    assert x.shape[1] == self.n_feats, f"BatchNorm1D requires input with {self.n_feats} features, got {x.shape[1]}"

    if self.train:
      mean = x.mean(axis=0)
      std  = x.std(axis=0, keepdims=True)
      x_norm = (x - mean) / (std + self.eps).sqrt()
      x_norm.name = "x_norm"

      self.running_mean = self.momentum * self.running_mean + (Tensor([1], requires_grad=False, name="1", device=self.device) - self.momentum) * mean
      self.running_var = self.momentum * self.running_var + (Tensor([1], requires_grad=False, name="1", device=self.device) - self.momentum) * std**2
    else:
      x_norm = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
      x_norm.name = "x_norm"

    self.out = self.weight * x_norm + self.bias
    return self.out


class BatchNorm2D(Layer):
  def __init__(self, n_feats: int, eps=1e-5, momentum=0.1):
    super().__init__()
    self.type = LayerType.BATCHNORM2D
    self.n_feats = n_feats
    self.eps      = Tensor([eps], requires_grad=False, name="batchnorm2d-eps", device=self.device)
    self.momentum = Tensor([momentum], requires_grad=False, name="batchnorm2d-momentum", device=self.device)

    self.weight = Tensor.ones((1, n_feats, 1, 1), name="batchnorm2d-gamma", device=self.device)
    self.bias   = Tensor.zeros((1, n_feats, 1, 1), name="batchnorm2d-beta", device=self.device)

    self.running_mean = Tensor.zeros((1, n_feats, 1, 1), requires_grad=False, name="batchnorm2d-running-mean", device=self.device)
    self.running_var  = Tensor.ones((1, n_feats, 1, 1), requires_grad=False, name="batchnorm2d-running-var", device=self.device)

    self.register_param("weight", self.weight)
    self.register_param("bias", self.bias)

  def __call__(self, x: Tensor) -> Tensor:
    assert len(x.shape) == 4, f"BatchNorm2D requires input of shape (N,C,H,W), got {x.shape}"
    N, C, H, W = x.shape
    assert C == self.n_feats, f"Expected {self.n_feats} channels, got {C}"

    if self.train:
      mean = x.mean(axis=(0, 2, 3), keepdims=True)   # shape (1,C,1,1)
      std  = x.std(axis=(0, 2, 3), keepdims=True)    # shape (1,C,1,1)
      x_norm = (x - mean) / (std + self.eps).sqrt()

      one = Tensor([1.0], requires_grad=False, name="1", device=self.device)
      self.running_mean = self.momentum * self.running_mean + (one - self.momentum) * mean
      self.running_var  = self.momentum * self.running_var  + (one - self.momentum) * std
    else:
      x_norm = (x - self.running_mean) / (self.running_var + self.eps).sqrt()

    x_norm.name = "x_norm"
    self.out = self.weight * x_norm + self.bias
    return self.out
