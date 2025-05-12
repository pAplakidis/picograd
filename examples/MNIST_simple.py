#!/usr/bin/env python3
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tqdm import tqdm


# TODO: find a way to import without this
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import picograd.nn as nn
from picograd.tensor import Tensor
from picograd.loss import *
from picograd.optim import *
from picograd.draw_utils import draw_dot

BS = 16

# FIXME: on CPU loss goes down, on GPU it goes up
device = Device(Devices.CUDA) if is_cuda_available() else Device(Devices.CPU)
# device = Device(Devices.CPU)
print("[*] Using device", device.name, "\n")

def get_data():
  (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
  X_train = X_train / 255.0
  X_test = X_test / 255.0
  return X_train, Y_train, X_test, Y_test


class Testnet(nn.Module):
  def __init__(self, in_feats, out_feats):
    super(Testnet, self).__init__()
    self.fc1 = nn.Linear(in_feats, 128)
    self.fc2 = nn.Linear(128, out_feats)

  def forward(self, x):
    x = self.fc1(x)
    x = x.relu()
    x = self.fc2(x)
    return x.softmax()


if __name__ == '__main__':
  X_train, Y_train, X_test, Y_test = get_data()

  model = Testnet(784, 10).to(device)
  # optim = Adam(model.get_params(), lr=1e-4)
  optim = SGD(model.get_params(), lr=1e-5)

  # Training Loop
  epochs = 4
  losses = []
  print("Training ...")
  for i in range(epochs):
    print(f"[+] Epoch {i+1}/{epochs}")
    epoch_losses = []

    num_batches = len(X_train) // BS + (len(X_train) % BS != 0)
    for batch_idx in (t := tqdm(range(num_batches), total=num_batches)):
      batch_start = batch_idx * BS
      batch_end = min(batch_start + BS, len(X_train))
      X_batch = X_train[batch_start:batch_end].reshape(-1, 784)
      Y_batch = Y_train[batch_start:batch_end]

      X = Tensor(np.array(X_batch, dtype=np.float32), device=device)
      Y = np.zeros((1, 10), dtype=np.float32)
      Y = np.zeros((len(Y_batch), 10), dtype=np.float32)
      for idx, label in enumerate(Y_batch):
        Y[idx][label] = 1.0
      Y = Tensor(Y, device=device)

      out = model(X)
      loss = CrossEntropyLoss(out, Y)
      losses.append(loss.data[0])
      epoch_losses.append(loss.mean().item)

      optim.zero_grad()
      loss.backward()
      optim.step()

      if batch_idx == 0 and i == 0: draw_dot(loss, path="graphs/mnist")
      t.set_description(f"Loss: {loss.mean().item:.2f}")

    print(f"Avg loss: {np.array(epoch_losses).mean()}")

  plt.plot(losses)
  plt.show()

  # Eval
  print("Evaluating ...")
  eval_losses = []
  for idx, x in (t := tqdm(enumerate(X_test), total=len(X_test))):
    X = Tensor(np.array([x], dtype=np.float32).reshape(-1, 784), device=device)
    Y = np.zeros((1, 10), dtype=np.float32)
    Y[0][Y_test[idx]] = 1.0
    Y = Tensor(Y, device=device)

    out = model(X)

    loss = CrossEntropyLoss(out, Y)
    eval_losses.append(loss.data[0])
    t.set_description(f"Loss: {loss.data[0]:.2f}")
  print(f"Avg loss: {np.array(eval_losses).mean()}")

  plt.plot(eval_losses)
  plt.show()

  # show results
  # for i in range(10):
  #   idx = random.randint(0, len(X_test))
  #   X = Tensor(np.array([X_test[idx]])).unsqueeze(0)
  #   Y = Y_test[idx]

  #   out = model(X)
  #   print(f"model: {np.argmax(out.data, axis=1)[0]} - GT: {Y}")
  #   plt.imshow(X_test[idx], cmap='gray')
  #   plt.show()

