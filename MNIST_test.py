#!/usr/bin/env python3
import numpy as np
import torchvision.datasets as datasets
import requests, gzip, os, hashlib, numpy

from tensor import Tensor
from loss import *
from optim import *
import nn

def get_data(use_dataset=False):
  if use_dataset:
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    return mnist_trainset

  def fetch(url):
    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
      with open(fp, "rb") as f:
        dat = f.read()
    else:
      with open(fp, "wb") as f:
        dat = requests.get(url).content
        f.write(dat)
    return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

  X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
  Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
  Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
  return X_train, Y_train, X_test, Y_test


# TODO: instead of manuallly passing the layers through ops, just forward them through a net module
class Testnet(nn.Module):
  def __init__(self, in_feats, out_feats):
    super(Testnet, self).__init__()
    self.dense1 = nn.Linear(in_feats, out_feats)

  def forward(self, x):
    x = self.dense1(x)
    return x


if __name__ == '__main__':
  X_train, Y_train, X_test, Y_test = get_data()

  print(X_train)
  exit(0)

  t_in = Tensor(np.random.rand(10), name="t_in")
  print("t_in:", t_in)
  #gt = Tensor(np.random.rand(1, 5), name="ground_truth")  # for regression
  #gt = Tensor(np.ones((1,5)) / 2, name="ground_truth")  # for classification
  gt = Tensor(np.ones((1,1)), name="ground_truth")  # for binary classification

  model = Testnet(t_in.shape([0]), 5)
  optim = SGD(model.get_params(), lr=1e-3)
  layer1 = nn.Linear(t_in.shape([0]), 5)
  layer2 = nn.Linear(5, 5)
  layer3 = nn.Linear(5, 1)

  lr = 1e-3

  # Training Loop
  epochs = 100
  for i in range(epochs):
    print("[+] epoch", i+1)
    t_in.layer = layer1
    t1 = layer1(t_in)
    t1.layer = layer2
    t2 = layer2(t1)
    t3 = t2.relu()
    t3.layer = layer3
    t4 = layer3(t3)   # for regression
    t5 = t4.softmax() # for classification

    #loss = MSELoss(t4, gt) # for regression
    #loss = CrossEntropyLoss(t5, gt) # for classification
    loss = BCELoss(t5, gt) # for binary classification
    print("loss:", loss.data)
    loss.backward()

    # manual optimization with SGD
    params = list(loss._prev.copy())
    params.insert(0, loss)
    params = list(reversed(params))

    params = manual_update(params, lr)
    params = reset_grad(params)

  print("\nNetwork Graph:")
  loss.print_graph()
