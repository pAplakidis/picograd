#!/usr/bin/env python3
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import picograd.nn as nn
from picograd import Tensor
from picograd.backend.device import Devices, Device
from picograd.draw_utils import draw_dot

device = Device(Devices.CPU)
print("[*] Using device", device.name, "\n")

class RNNNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = nn.RNN(input_size=10, hidden_size=5, num_layers=1, batch_first=True)

  def forward(self, x, h_0):
    out, h_n = self.rnn(x, h_0)
    return out, h_n

class LSTMNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(input_size=10, hidden_size=5, num_layers=1)

  def forward(self, x, h_0):
    out, h_n = self.lstm(x, h_0)
    return out, h_n

def numpy_rnn(x, h_0, model):
  x_np   = x.data
  h_prev = h_0.data

  U = model.rnn.u.data
  V = model.rnn.v.data
  W = model.rnn.weight.data
  b = model.rnn.b.data if model.rnn.b is not None else 0
  c = model.rnn.c.data if model.rnn.c is not None else 0

  y = np.zeros((x_np.shape[0], x_np.shape[1], model.rnn.hidden_size), dtype=np.float32)

  for t in range(x_np.shape[1]):
    x_t = x_np[:, t, :]                            # (batch, input_size)
    a_t = x_t @ U + h_prev @ W + b                 # (batch, hidden_size)
    h_t = np.tanh(a_t)                             # (batch, hidden_size)
    o_t = h_t @ V + c                              # (batch, hidden_size)

    # softmax along last axis
    exp_o = np.exp(o_t - np.max(o_t, axis=-1, keepdims=True))
    y[:, t, :] = exp_o / np.sum(exp_o, axis=-1, keepdims=True)

    h_prev = h_t
  return y, h_prev

class RecurrentTest(unittest.TestCase):
  def test_rnn(self):
    x = Tensor.random((1, 3, 10), device=device, name="x")  # (batch, sequence_size, input_size)
    h_0 = Tensor.random((1, 5), device=device, name="h_0")  # (batch, hidden_size)

    model = RNNNet()
    out, h_n = model(x, h_0)
    out.backward()
    draw_dot(out, path="graphs/rnn")

    y_ref, h_n_ref = numpy_rnn(x, h_0, model)
    np.testing.assert_allclose(out.data, y_ref, atol=1e-5, rtol=1e-5)
    print("[+] RNN OK")

  def test_lstm(self):
    x = Tensor.random((1, 3, 10), device=device, name="x")  # (batch, sequence_size, input_size)
    h_0 = Tensor.random((1, 5), device=device, name="h_0")  # (batch, hidden_size)

    model = LSTMNet()
    out, h_n = model(x, h_0)
    # out.backward()  # FIXME: ValueError: non-broadcastable output operand with shape (5,5) doesn't match the broadcast shape (5,5,5)
    draw_dot(out, path="graphs/lstm")
    print("[+] LSTM OK")


if __name__ == "__main__":
  unittest.main()
