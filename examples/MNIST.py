#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tqdm import tqdm

# TODO: find a way to import without this (install module)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import picograd as pg
import picograd.nn as nn
from picograd.optim import AdamW
from picograd.loss import CrossEntropyLoss

BS = 16
LR = 1e-6
EPOCHS = 30

device = pg.Device(pg.Devices.CPU)
# device = pg.Device(pg.Devices.CUDA) if pg.is_cuda_available() else pg.Device(pg.Devices.CPU)
print("[*] Using device", device.name, "\n")

def get_data():
  (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
  X_train = X_train / 255.0
  X_test = X_test / 255.0
  return X_train, Y_train, X_test, Y_test


class Testnet(nn.Module):
  def __init__(self, out_feats):
    super(Testnet, self).__init__()
    self.conv2d = nn.Conv2D(1, 1, 3)
    self.bn2d = nn.BatchNorm2D(1)
    self.pool = nn.MaxPool2D()
    # self.fc = nn.Linear(625, out_feats)
    self.fc = nn.Linear(784, out_feats)

  def forward(self, x):
    # x = self.pool(self.bn2d(self.conv2d(x))).relu()
    x = x.reshape(x.shape[0], -1) # TODO: use flatten when done fixing it
    x = self.fc(x)
    return x.softmax()


if __name__ == '__main__':
  X_train, Y_train, X_test, Y_test = get_data()

  in_feats = X_train.shape[1] * X_train.shape[2]
  model = Testnet(10).to(device)
  optim = AdamW(model.get_params(), lr=LR)

  # Training Loop
  losses = []
  accuracies = []
  print("Training ...")
  # model.train_mode()
  for i in range(EPOCHS):
    print(f"[=>] Epoch {i+1}/{EPOCHS}")
    epoch_losses = []
    epoch_accs = []

    num_batches = len(X_train) // BS + (len(X_train) % BS != 0)
    for batch_idx in (t := tqdm(range(num_batches), total=num_batches)):
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

      acc = (np.argmax(out.data, axis=1) == Y.data).mean()
      accuracies.append(acc)
      epoch_accs.append(acc)

      optim.zero_grad()
      loss.backward()
      optim.step()

      if batch_idx == 0 and i == 0: pg.draw_dot(loss, path="graphs/mnist")
      t.set_description(f"Loss: {loss.mean().item:.2f} - Acc: {acc:.2f}")
    print(f"Avg loss: {np.array(epoch_losses).mean():.4f} - Avg acc: {np.array(epoch_accs).mean():.2f}")

  plt.plot(losses)
  plt.plot(accuracies)
  plt.show()

  # Eval
  print("Evaluating ...")
  # model.eval_mode()
  eval_losses = []
  eval_accs = []
  num_batches = len(X_train) // BS + (len(X_train) % BS != 0)
  for batch_idx in (t := tqdm(range(num_batches), total=num_batches)):
    batch_start = batch_idx * BS
    batch_end = min(batch_start + BS, len(X_train))
    X_batch = np.expand_dims(X_train[batch_start:batch_end], axis=1)
    Y_batch = Y_train[batch_start:batch_end]

    X = pg.Tensor(np.array(X_batch), device=device)
    Y = pg.Tensor(np.array(Y_batch), device=device)

    out = model(X)
    loss = CrossEntropyLoss(out, Y)
    eval_losses.append(loss.data[0])

    acc = (np.argmax(out.data, axis=1) == Y.data).mean()
    eval_accs.append(acc)

    t.set_description(f"Loss: {loss.data[0]:.2f} - Acc: {acc:.2f}")
  print(f"Avg loss: {np.array(eval_losses).mean()} - Avg acc: {np.array(eval_accs).mean():.2f}")

  plt.plot(eval_losses)
  plt.plot(eval_accs)
  plt.show()

  # show results
  for i in range(10):
    idx = random.randint(0, len(X_test))
    X_batch = np.expand_dims([X_test[idx]], axis=1)
    X = pg.Tensor(np.array(X_batch), device=device)
    Y = Y_test[idx]

    out = model(X)
    print(f"model: {np.argmax(out.data, axis=1)[0]} - GT: {Y}")
    plt.imshow(X_test[idx], cmap='gray')
    plt.show()
