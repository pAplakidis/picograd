#!/usr/bin/env python3
import os
import sys
import unittest
import numpy as np

# TODO: find a way to import without this (install module)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import picograd as pg
import picograd.nn as nn
from picograd.optim import Adam
from picograd.loss import CrossEntropyLoss

BS = 16

device = pg.Device(pg.Devices.CPU)
# device = pg.Device(pg.Devices.CUDA) if pg.is_cuda_available() else pg.Device(pg.Devices.CPU)
print("[*] Using device", device.name, "\n")

# generate mock data like MNIST
def get_data(num_train=100, num_test=20):
  # Fake MNIST images: values in [0,1] after normalization
  X_train = np.random.rand(num_train, 28, 28).astype("float32")
  Y_train = np.random.randint(0, 10, size=(num_train,), dtype="int64")
  
  X_test = np.random.rand(num_test, 28, 28).astype("float32")
  Y_test = np.random.randint(0, 10, size=(num_test,), dtype="int64")
  
  return X_train, Y_train, X_test, Y_test


class ConvNet(nn.Module):
  def __init__(self, out_feats):
    super(ConvNet, self).__init__()
    self.conv2d = nn.Conv2D(1, 1, 3)
    self.bn2d = nn.BatchNorm2D(1)
    self.pool = nn.MaxPool2D()
    self.fc = nn.Linear(625, out_feats)

  def forward(self, x):
    x = self.pool(self.bn2d(self.conv2d(x))).relu()
    x = x.reshape(BS, -1) # TODO: use flatten when done fixing it
    x = self.fc(x)
    return x.softmax()


class ConvNetIntegrationTest(unittest.TestCase):
  def test_module(self):
    X_train, Y_train, X_test, Y_test = get_data()
    in_feats = X_train.shape[1] * X_train.shape[2]
    model = ConvNet(10).to(device)
    optim = Adam(model.get_params(), lr=1e-5)

    # Training Loop
    epochs = 1
    losses = []
    print("Training ...")
    # model.train_mode()
    for i in range(epochs):
      print(f"[=>] Epoch {i+1}/{epochs}")
      epoch_losses = []

      num_batches = len(X_train) // BS + (len(X_train) % BS != 0)
      for batch_idx in range(num_batches):
        batch_start = batch_idx * BS
        batch_end = min(batch_start + BS, len(X_train))
        X_batch = np.expand_dims(X_train[batch_start:batch_end], axis=1)
        Y_batch = Y_train[batch_start:batch_end]
        X = pg.Tensor(np.array(X_batch), device=device)
        Y = pg.Tensor(np.array(Y_batch), device=device)

        out = model(X)
        loss = CrossEntropyLoss(out, Y)
        losses.append(loss.data[0])
        epoch_losses.append(loss.mean().item)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"Loss: {loss.mean().item:.2f}")
        break
      print(f"Avg loss: {np.array(epoch_losses).mean()}")

    # Eval
    print("Evaluating ...")
    # model.eval_mode()
    eval_losses = []
    num_batches = len(X_train) // BS + (len(X_train) % BS != 0)
    for batch_idx in range(num_batches):
      batch_start = batch_idx * BS
      batch_end = min(batch_start + BS, len(X_train))
      X_batch = np.expand_dims(X_train[batch_start:batch_end], axis=1)
      Y_batch = Y_train[batch_start:batch_end]
      X = pg.Tensor(np.array(X_batch), device=device)
      Y = pg.Tensor(np.array(Y_batch), device=device)

      out = model(X)
      loss = CrossEntropyLoss(out, Y)
      eval_losses.append(loss.data[0])
      print(f"Loss: {loss.data[0]:.2f}")
      break
    print(f"Avg loss: {np.array(eval_losses).mean()}")

    print("[+] ConvNet Module Integration Test OK")


if __name__ == "__main__":
  unittest.main()
