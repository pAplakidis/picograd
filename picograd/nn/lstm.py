from .module import Layer, LayerType
from picograd.tensor import Tensor


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
