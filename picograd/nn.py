#!/usr/bin/env python3
import numpy as np
from enum import Enum, auto
from picograd.tensor import Tensor
from picograd.backend.cpu.ops import *
from picograd.backend.device import Devices, Device

class LayerType(Enum):
  NOLAYER = auto()
  LINEAR = auto()
  CONV2D = auto()

  MAXPOOL2D = auto()
  AVGPOOL2D = auto()

  BATCHNORM2D = auto()
  LAYERNORM = auto()

  RNN = auto()

  def __str__(self):
    return self.name

# TODO: Sequential

class Module:
  def __init__(self):
    self.params = []
    self.train = True
    self.device = Device(Devices.CPU)

  def to(self, device: Device):
    self.device = device
    for param in self.get_params(): param.to(device)
    return self

  def train_mode(self):
    self.train = True
    for param in self.get_params(): param.train = True
    return self

  def eval_mode(self):
    self.train = False
    for param in self.get_params(): param.train = False
    return self

  def forward(self):
    return None

  def __call__(self, *params):
    return self.forward(*params)

  def get_params(self):
    # TODO: params are tensors(weights, biases, etc) not layers
    self.params = []
    for name, param in self.__dict__.items():
      if isinstance(param, Layer):
        self.params.append(param)
    return self.params


# TODO: for all Tensors defined inside the layers, set device to self.device
class Layer:
  def __init__(self, device = Device(Devices.CPU)):
    self.type = None
    self.t_in = None
    self.t_out = None
    self.weight = None
    self.bias = None
    self.train = True
    self.device = device

    self.subgraph_name = None
    self._subgraph_nodes = []

  def to(self, device: Device):
    self.device = device
    if self.weight is not None: self.weight.to(device)
    if self.bias is not None: self.bias.to(device)
    return self

  def _track(self, t: Tensor):
    """Tracks tensors that belong to this layer's subgraph for better visualization"""
    self._subgraph_nodes.append(t)
    t.layer = self
    return t

class Linear(Layer):
  def __init__(self, in_feats: int, out_feats: int, initialization: str = 'gaussian'):
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
    self.bias = Tensor(np.zeros((self.out_feats,)), name="linear-bias", device=self.device)

  def __call__(self, x: Tensor):
    assert len(x.shape) >= 2, "Input Tensor requires batch_size dimension"
    self.t_in = x
    self.t_out = x.linear(self.weight, self.bias)
    return self.t_out


class Conv2d(Layer):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0):
    super().__init__()
    self.type = LayerType.CONV2D
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    self.weight = Tensor(np.random.uniform(0.0, 1.0, (out_channels, in_channels, kernel_size, kernel_size)), "conv2D_kernel", device=self.device)
    self.bias = Tensor(np.zeros((out_channels,)), name="bias", device=self.device)

    assert self.kernel_size % 2 != 0, "Conv2D kenrel_size must be odd"
    assert self.kernel_size in [3, 5, 7, 9]

  def __call__(self, x: Tensor):
    self.t_in = x
    self.t_out = x.conv2d(self.weight, self.bias, self.in_channels, self.out_channels, self.stride, self.padding, debug=False)
    return self.t_out


class MaxPool2D(Layer):
  def __init__(self, filter=(2,2), stride=1):
    super().__init__()
    self.type = LayerType.MAXPOOL2D
    
    self.filter = filter
    self.stride = stride

  def __call__(self, x: Tensor):
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

  def __call__(self, x: Tensor):
    self.t_in = x
    self.t_out = x.avgpool2d(self.filter, self.stride, self.padding)
    return self.t_out

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

  def __call__(self, x: Tensor):
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

  def __call__(self, x: Tensor):
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


class LayerNorm(Layer):
  def __init__(self, normalized_shape: Tuple[int], eps=1e-5):
    super().__init__()
    self.type = LayerType.LAYERNORM
    if isinstance(normalized_shape, int): normalized_shape = (normalized_shape,)
    self.normalized_shape = tuple(normalized_shape)
    self.eps = Tensor([eps], requires_grad=False, name="layernorm-eps", device=self.device)

    self.weight = Tensor.ones(self.normalized_shape, name="layernorm-gamma", device=self.device)
    self.bias   = Tensor.zeros(self.normalized_shape, name="layernorm-beta", device=self.device)

  def __call__(self, x: Tensor):
    axes = tuple(range(-len(self.normalized_shape), 0)) # normalize across last len(normalized_shape) dims
    mean = x.mean(axis=axes, keepdims=True)
    std  = x.std(axis=axes, keepdims=True)
    x_norm = (x - mean) / (std+ self.eps).sqrt()
    x_norm.name = "x_norm"
    self.out = self.weight * x_norm + self.bias
    return self.out


class RNN(Layer):
  def __init__(
      self,
      input_size: int,
      hidden_size: int,
      num_layers=1,
      nonlinearity='tanh',
      bias=True,
      batch_first=False,
      dropout=0.0,
      bidirectional=False
  ):
    super().__init__()
    self.type = LayerType.RNN

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.batch_first = batch_first
    self.nonlinearity = nonlinearity
    self.dropout = dropout
    self.bidirectional = bidirectional
    
    # FIXME: optim - learnable paramaters are no longer weight and bias (add to a params[] list and modify optim)
    self.u      = Tensor.random((input_size, hidden_size), device=self.device, name="rnn_u")  # input to hidden weight
    self.v      = Tensor.random((hidden_size, hidden_size), device=self.device, name="rnn_v") # hidden to output weight
    self.weight = Tensor.random((hidden_size, hidden_size), device=self.device, name="rnn_w") # hidden to hidden weight
    self.b      = Tensor.zeros((hidden_size,), device=self.device, name="rnn_b") if bias else None
    self.c      = Tensor.zeros((hidden_size,), device=self.device, name="rnn_c") if bias else None

  def __call__(self, x: Tensor, h_0: Tensor=None):
    """"""
    assert x.shape == (x.shape[0], x.shape[1], self.input_size), f"Expected input shape (batch_size, seq_len, {self.input_size}), got {x.shape}"

    y = Tensor.zeros((x.shape[0], x.shape[1], self.hidden_size), device=self.device, name="rnn_out") # output tensor
    h_prev = Tensor.zeros((x.shape[0], self.hidden_size), device=self.device, name="h_0") if h_0 is None else h_0
    # FIXME: graph is wrong
    for t in range(x.shape[1]):
      x_t = x[:, t, :]                  # (batch, input_size)
      a_t = x_t @ self.u + h_prev @ self.weight + self.b
      h_t = a_t.tanh()
      o_t = h_t @ self.v + self.c       # (batch, hidden_size)
      y[:, t, :] = o_t.softmax(axis=-1)
      h_prev = h_t

    return y, h_prev
