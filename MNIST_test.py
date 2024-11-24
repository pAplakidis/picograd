#!/usr/bin/env python3
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from keras.datasets import mnist
from tqdm import tqdm

from tensor import Tensor
from loss import *
from optim import *
import nn

def get_data():
  (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
  return X_train, Y_train, X_test, Y_test


class Testnet(nn.Module):
  def __init__(self, in_feats, out_feats):
    super(Testnet, self).__init__()
    self.dense1 = nn.Linear(in_feats, 128)
    self.dense2 = nn.Linear(128, out_feats)

  def forward(self, x):
    x = self.dense1(x)
    x = x.relu()
    x = self.dense2(x)
    return x.softmax()


if __name__ == '__main__':
  X_train, Y_train, X_test, Y_test = get_data()

  in_feats = X_train.shape[1] * X_train.shape[2]
  model = Testnet(in_feats, 10)
  optim = SGD(model.get_params(), lr=1e-4)

  # Training Loop
  epochs = 1
  losses = []
  print("Training ...")
  for i in range(epochs):
    print(f"[+] Epoch {i+1}/{epochs}")
    epoch_losses = []
    for idx, x in (t := tqdm(enumerate(X_train), total=len(X_train))):
      t_in = Tensor(np.array([x])).flatten().unsqueeze(0)
      Y = np.zeros((1, 10), dtype=np.float32)
      Y[0][Y_train[idx]] = 1.0
      Y = Tensor(Y)

      out = model(t_in)

      loss = CrossEntropyLoss(out, Y)
      losses.append(loss.data[0])
      epoch_losses.append(loss.data[0])
      t.set_description(f"Loss: {loss.data[0]:.2f}")

      optim.reset_grad()
      loss.backward()
      optim.step()
    print(f"Avg loss: {np.array(epoch_losses).mean()}")

  plt.plot(losses)
  plt.show()

  # Eval
  print("Evaluating ...")
  eval_losses = []
  for idx, x in (t := tqdm(enumerate(X_test), total=len(X_test))):
    t_in = Tensor(np.array([x])).flatten().unsqueeze(0)
    Y = np.zeros((1, 10), dtype=np.float32)
    Y[0][Y_test[idx]] = 1.0
    Y = Tensor(Y)

    out = model(t_in)

    loss = CrossEntropyLoss(out, Y)
    eval_losses.append(loss.data[0])
    t.set_description(f"Loss: {loss.data[0]:.2f}")
  print(f"Avg loss: {np.array(eval_losses).mean()}")

  plt.plot(eval_losses)
  plt.show()

  # show results
  for i in range(10):
    idx = random.randint(0, len(X_test))
    X = X_test[idx]
    t_in = Tensor(np.array([X])).flatten().unsqueeze(0)
    Y = Y_test[idx]

    out = model(t_in)
    print(f"model: {np.argmax(out.data, axis=1)[0]} - GT: {Y}")
