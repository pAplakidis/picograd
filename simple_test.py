#!/usr/bin/env python3
import numpy as np

from tensor import Tensor
from loss import *
from optim import *
import nn

# TODO: instead of manuallly passing the layers through ops, just forward them through a net module
class Testnet(nn.Module):
  def __init__(self, in_feats, out_feats):
    super(Testnet, self).__init__()
    self.dense1 = nn.Linear(in_feats, out_feats)

  def forward(self, x):
    x = self.dense1(x)
    return x


if __name__ == '__main__':
  t_in = Tensor(np.random.rand(10), name="t_in")
  print("t_in:", t_in)
  #gt = Tensor(np.random.rand(1, 5), name="ground_truth")  # for regression
  gt = Tensor(np.ones((1,5)) / 2, name="ground_truth")  # for classification

  model = Testnet(t_in.shape([0]), 5)
  optim = SGD(model.get_params(), lr=1e-3)
  layer1 = nn.Linear(t_in.shape([0]), 5)
  layer2 = nn.Linear(5, 5)
  layer3 = nn.Linear(5, 5)

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
    loss = CrossEntropyLoss(t4, gt) # for classification
    
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
