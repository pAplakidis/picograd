#!/usr/bin/env python3
import numpy as np

from tensor import Tensor
from loss import MSELoss
from optim import SGD
import nn

# TODO: instead of manuallly passing the layers through ops, just forward them through a net module
class Testnet(nn.Module):
  def __init__(self, in_feats, out_feats):
    super(Testnet, self).__init__()
    self.dense1 = nn.Linear(in_feats, out_feats)

  def forward(self, x):
    x = self.dense1(x)
    return x


def manual_update(params):
  for i in range(len(params)):
    if params[i].layer != None:
      if params[i].w != None:
        #params[i].w += -lr * params[i].w.grad
        # NOTE: using lr decreases loss while -lr increases it, BUT -lr is the correct one
        params[i].w += lr * params[i].w.grad
        params[i].layer.weight = params[i].w
      if params[i].b != None:
        #params[i].b += -lr * params[i].b.grad
        params[i].b += lr * params[i].b.grad
        params[i].layer.bias = params[i].b

  return params

def reset_grad(params):
  for i in range(len(params)):
    params[i].grad = np.ones_like(params[i].grad)
  return params


if __name__ == '__main__':
  t_in = Tensor(np.random.rand(10), name="t_in")
  print("t_in:", t_in)
  gt = Tensor(np.random.rand(1, 5), name="ground_truth")

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
    t2.layer = layer3
    t3 = layer3(t2)

    loss = MSELoss(t2, gt)
    print("loss:", loss.data)
    loss.backward()


    # manual optimization with SGD
    print()
    params = list(loss._prev.copy())
    params.insert(0, loss)
    params = list(reversed(params))

    params = manual_update(params)
    params = reset_grad(params)

  t3.print_graph()  # BUG: in _prev order when adding 2 layers (t_in is in the wrong place)
