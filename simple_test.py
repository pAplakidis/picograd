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


if __name__ == '__main__':
  t_in = Tensor(np.random.rand(10), name="t_in")
  print("t_in:", t_in)
  perceptron = nn.Linear(t_in.shape([0]), 5)
  t_out = perceptron(t_in)
  print("t_out:", t_out)
  #print()
  #print("==Backpropagation of t_out==")
  #t_out.backward()

  gt = Tensor(np.random.rand(1, 5), name="ground_truth")
  loss = MSELoss(t_out, gt)
  print("loss:", loss)
  print()
  #for p in loss._prev:
  #  print("[*] ", p)
  print("==Backpropagation of Loss==")
  loss.backward()

  #optim = SGD(lr=1e-3)  # TODO: once a net obj is fully functional, pass it to the optimizer

  # manual optimization with SGD
  print()
  tmp = list(loss._prev.copy())
  tmp.insert(0, loss)
  tmp = reversed(tmp)
  lr = 1e-3

  # TODO: update the params of the layers not the tensors (means i need to change things)
  for t in tmp:
    if t.w != None:
      print("Updating weights of:", t, "...")
      t.w += -lr * t.w.grad
    if t.b != None:
      print("Updating biases of:", t, "...")
      t.b += -lr * t.b.grad
  
  print("Net after SGD step")
  loss.print_graph()
