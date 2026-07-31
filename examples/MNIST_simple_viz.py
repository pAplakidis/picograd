#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import picograd.nn as nn
from picograd.tensor import Tensor, default_device
from picograd.viz import enabled as viz_enabled, flush as viz_flush, trace_path


class Testnet(nn.Module):
  def __init__(self, in_feats, out_feats):
    super(Testnet, self).__init__()
    self.fc1 = nn.Linear(in_feats, 128)
    self.fc2 = nn.Linear(128, out_feats)

  def forward(self, x):
    x = self.fc1(x)
    x = x.relu()
    x = self.fc2(x)
    return x.softmax(axis=-1)


if __name__ == "__main__":
  np.random.seed(0)
  device = default_device()
  print("[*] Using device", device.name)

  model = Testnet(784, 10).to(device)
  x = Tensor(np.random.rand(1, 784).astype(np.float32), name="mnist-sample", requires_grad=False, device=device)

  out = model(x)
  probs = out.numpy()
  print("prediction:", int(np.argmax(probs, axis=1)[0]))
  print("probabilities:", probs)

  if viz_enabled():
    path = viz_flush()
    print("viz trace:", path)
  else:
    print("viz disabled: run with VIZ=1 to write", trace_path())
