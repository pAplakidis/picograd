#!/usr/bin/env python3
import numpy as np

from tensor import Tensor
from loss import MSELoss
from optim import SGD
import nn

# TODO: instead of manuallly passing the layers through ops, just forward them through a net module
class Testnet(nn.Module):
  def __init__(self):
    pass

  def forward(self, x):
    return x

def manual_update(params):
  for i in range(len(params)):
    if params[i].layer != None:
      if params[i].w != None:
        #print("Updating weights of:", t, "...")
        params[i].w += -lr * params[i].w.grad
        params[i].layer.weight = params[i].w
      if params[i].b != None:
        #print("Updating biases of:", t, "...")
        params[i].b += -lr * params[i].b.grad
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

  perceptron = nn.Linear(t_in.shape([0]), 5)
  t_in.layer = perceptron
  lr = 1e-4

  # BUG: Loss goes up instead of decreasing
  # Training Loop
  epochs = 100
  for i in range(epochs):
    print("epoch", i+1)
    t_out = perceptron(t_in)
    #print("t_out:", t_out)
    #print()
    #print("==Backpropagation of t_out==")
    #t_out.backward()

    loss = MSELoss(t_out, gt)
    print("loss:", loss.data)
    print()
    #for p in loss._prev:
    #  print("[*] ", p)
    #print("==Backpropagation of Loss==")
    loss.backward()

    #optim = SGD(lr=1e-3)  # TODO: once a net obj is fully functional, pass it to the optimizer

    # manual optimization with SGD
    print()
    params = list(loss._prev.copy())
    params.insert(0, loss)
    params = list(reversed(params))

    # TODO: update the params of the layers not the tensors (means i need to change things)
    # TODO: implement a full training loop
    params = manual_update(params)
    #for p in params:
    #  p.print_graph()
    params = reset_grad(params)

  #print("Net after SGD step")
  #loss.print_graph()
