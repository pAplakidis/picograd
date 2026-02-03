from .module import Layer, LayerType
from picograd.tensor import Tensor


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
