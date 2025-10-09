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
  LSTM = auto()

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

  def get_layers(self):
    layers = []
    for _, v in self.__dict__.items():
      if isinstance(v, Layer):
        layers.append(v)
    return layers

  def get_params(self):
    params = []
    for layer in self.get_layers():
      params.extend(layer.parameters())
    return params


class Layer:
  def __init__(self, device = Device(Devices.CPU)):
    self.type = None
    self.t_in = None
    self.t_out = None
    self.train = True
    self.device = device
    self._params: dict[str, Tensor] = {}

    self.subgraph_name = None
    self._subgraph_nodes = []

  def register_param(self, name: str, tensor: Tensor):
    tensor.name = name
    self._params[name] = tensor
    return tensor

  def parameters(self):
    """Returns a list of Tensors registered on this layer."""
    return list(self._params.values())

  def to(self, device: Device):
    self.device = device
    for param in self.parameters(): param.to(device)
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

    self.register_param("weight", self.weight)
    self.register_param("bias", self.bias)

  def __call__(self, x: Tensor):
    assert len(x.shape) >= 2, "Input Tensor requires batch_size dimension"
    self.t_in = x
    self.t_out = x.linear(self.weight, self.bias)
    return self.t_out


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

    self.register_param("weight", self.weight)
    self.register_param("bias", self.bias)

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

    self.register_param("weight", self.weight)
    self.register_param("bias", self.bias)

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

    self.register_param("weight", self.weight)
    self.register_param("bias", self.bias)

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
    
    # TODO: use these
    self.batch_first = batch_first
    self.nonlinearity = nonlinearity
    self.dropout = dropout
    self.bidirectional = bidirectional
    
    self.u = self.register_param('u',           Tensor.random((input_size, hidden_size), device=self.device, name="rnn_u"))
    self.v = self.register_param('v',           Tensor.random((hidden_size, hidden_size), device=self.device, name="rnn_v"))
    self.weight = self.register_param("weight", Tensor.random((hidden_size, hidden_size), device=self.device, name="rnn_w"))

    if bias:
      self.b = self.register_param('b', Tensor.zeros((hidden_size,), device=self.device, name="rnn_b"))
      self.c = self.register_param('c', Tensor.zeros((hidden_size,), device=self.device, name="rnn_c"))
    else:
      self.b = None
      self.c = None

  def __call__(self, x: Tensor, h_0: Tensor=None):
    """"""
    assert x.shape == (x.shape[0], x.shape[1], self.input_size), f"Expected input shape (batch_size, seq_len, {self.input_size}), got {x.shape}"

    y      = Tensor.zeros((x.shape[0], x.shape[1], self.hidden_size), device=self.device, name="rnn_out")
    h_prev = Tensor.zeros((x.shape[0], self.hidden_size), device=self.device, name="h_0") if h_0 is None else h_0

    # TODO: cleaner solution graph is wrong due to slicing
    # for t in range(x.shape[1]):
    #   x_t = x[:, t, :]                  # (batch, input_size)
    #   a_t = x_t @ self.u + h_prev @ self.weight + self.b
    #   h_t = a_t.tanh()
    #   o_t = h_t @ self.v + self.c       # (batch, hidden_size)
    #   y[:, t, :] = o_t.softmax(axis=-1)
    #   h_prev = h_t

    ys = []
    xs = [x[:, t, :] for t in range(x.shape[1])]
    for x_t in xs:
      a_t = x_t @ self.u + h_prev @ self.weight + self.b
      h_t = a_t.tanh()
      o_t = h_t @ self.v + self.c
      y_t = o_t.softmax()
      ys.append(y_t)
      h_prev = h_t

    y = Tensor.stack(ys, axis=1)  # (batch, seq_len, hidden_size)
    return y, h_prev


class LSTM(Layer):
  def __init__(
      self,
      input_size: int,
      hidden_size: int,
      num_layers=1,
      bias=True,
      dropout=0.0,
      bidirectional=False,
      proj_size=0
  ):
    super().__init__()
    self.type = LayerType.LSTM

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    # TODO: use these
    self.dropout = dropout
    self.bidirectional = bidirectional
    self.proj_size = proj_size

    # input weights
    self.w_i  = self.register_param("w_i", Tensor.random((input_size, hidden_size), device=self.device, name="lstm_w_i"))
    self.w_f  = self.register_param("w_f", Tensor.random((input_size, hidden_size), device=self.device, name="lstm_w_f"))
    self.w_o  = self.register_param("w_o", Tensor.random((input_size, hidden_size), device=self.device, name="lstm_w_o"))
    self.w_c  = self.register_param("w_c", Tensor.random((input_size, hidden_size), device=self.device, name="lstm_w_c"))

    # hidden weights
    self.w_hi = self.register_param("w_hi", Tensor.random((hidden_size, hidden_size), device=self.device, name="lstm_w_hi"))
    self.w_hf = self.register_param("w_hf", Tensor.random((hidden_size, hidden_size), device=self.device, name="lstm_w_hf"))
    self.w_ho = self.register_param("w_ho", Tensor.random((hidden_size, hidden_size), device=self.device, name="lstm_w_ho"))
    self.w_hc = self.register_param("w_hc", Tensor.random((hidden_size, hidden_size), device=self.device, name="lstm_w_ho"))

    if bias:
      self.b_i = self.register_param("b_i", Tensor.zeros((hidden_size,), device=self.device, name="lstm_b_i"))
      self.b_f = self.register_param("b_f", Tensor.zeros((hidden_size,), device=self.device, name="lstm_b_f"))
      self.b_o = self.register_param("b_o", Tensor.zeros((hidden_size,), device=self.device, name="lstm_b_o"))
      self.b_c = self.register_param("b_c", Tensor.zeros((hidden_size,), device=self.device, name="lstm_b_c"))
    else:
      self.b_i = self.b_f = self.b_o = self.b_c = None

  def __call__(self, x: Tensor, h_0: Tensor=None, c_0: Tensor=None):
    """
    x: (batch, seq_len, input_size)
    h_0: (batch, hidden_size)
    c_0: (batch, hidden_size)
    """
    assert x.shape == (x.shape[0], x.shape[1], self.input_size), f"Expected input shape (batch_size, seq_len, {self.input_size}), got {x.shape}"
    assert x.shape[2] == self.input_size, f"Expected input_size={self.input_size}, got {x.shape[2]}"

    batch_size, seq_len, _ = x.shape
    h_prev = Tensor.zeros((batch_size, self.hidden_size), device=self.device) if h_0 is None else h_0
    c_prev = Tensor.zeros((batch_size, self.hidden_size), device=self.device) if c_0 is None else c_0

    y      = Tensor.zeros((batch_size, seq_len, self.hidden_size), device=self.device, name="rnn_out")
    h_prev = Tensor.zeros((batch_size, self.hidden_size), device=self.device, name="h_0") if h_0 is None else h_0

    ys = []
    xs = [x[:, t, :] for t in range(seq_len)]
    for x_t in xs:
      i_t = (x_t @ self.w_i + h_prev @ self.w_hi + self.b_i).sigmoid()
      f_t = (x_t @ self.w_f + h_prev @ self.w_hf + self.b_f).sigmoid()
      o_t = (x_t @ self.w_o + h_prev @ self.w_ho + self.b_o).sigmoid()
      g_t = (x_t @ self.w_c + h_prev @ self.w_hc + self.b_c).tanh()

      c_t = f_t * c_prev + i_t * g_t
      h_t = o_t * c_t.tanh()
      ys.append(h_t)
      h_prev, c_prev = h_t, c_t

    y = Tensor.stack(ys, axis=1)  # (batch, seq_len, hidden_size)
    return y, h_prev
