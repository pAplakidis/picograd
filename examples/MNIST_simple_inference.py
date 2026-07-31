#!/usr/bin/env python3
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MNIST_simple import Testnet
from picograd.nn.state import load
from picograd.tensor import Tensor, default_device

N_SAMPLES = 9

if __name__ == "__main__":
  device = default_device()
  print("[*] Using device", device.name)

  (_, _), (X_test, Y_test) = mnist.load_data()
  X_test = X_test / 255.0

  model = Testnet(784, 10).to(device)
  model.eval_mode()
  model.load_state_dict(load("./checkpoints/mnist_simple.pth"))
  print("[+] Loaded checkpoints/mnist_simple.pth")

  idxs = np.random.RandomState(0).choice(len(X_test), N_SAMPLES, replace=False)
  X = Tensor(np.array(X_test[idxs].reshape(N_SAMPLES, -1), dtype=np.float32), device=device)
  Y = Y_test[idxs]

  out = model(X)
  preds = np.argmax(out.numpy(), axis=1)

  assert preds.shape == (N_SAMPLES,)
  assert np.all((preds >= 0) & (preds <= 9))
  for i in range(N_SAMPLES):
    mark = "correct" if preds[i] == Y[i] else "WRONG"
    print(f"sample {i}: pred={preds[i]} gt={Y[i]} ({mark})")
  acc = (preds == Y).mean()
  print(f"Accuracy on {N_SAMPLES} samples: {acc:.2f}")

  fig, axes = plt.subplots(3, 3, figsize=(8, 8))
  for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[idxs[i]], cmap="gray")
    color = "green" if preds[i] == Y[i] else "red"
    ax.set_title(f"pred: {preds[i]} gt: {Y[i]}", color=color)
    ax.axis("off")
  fig.tight_layout()
  plt.show()
