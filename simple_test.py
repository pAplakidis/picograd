#!/usr/bin/env python3
import numpy as np

from tensor import Tensor
from loss import MSELoss
from nn import *


if __name__ == '__main__':
  t_in = Tensor(np.random.rand(10), name="t_in")
  print("t_in:", t_in)
  #print(t_in._prev)
  perceptron = Linear(t_in.shape([0]), 5)
  t_out = perceptron(t_in)
  print("t_out:", t_out)
  t_out.backward()
  gt = Tensor(np.random.rand(1, 5), name="gt")
  loss = MSELoss(t_out, gt)
  print("loss:", loss)
  print(loss._prev)
  loss.backward()
  # TODO: backpropagate fully
