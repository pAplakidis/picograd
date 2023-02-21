#!/usr/bin/env python3
import numpy as np

from tensor import Tensor
from loss import MSELoss
from nn import *

from graphviz import Digraph

# TODO: implement graph visualization
def trace():
  pass

def graph():
  pass

def draw_dot():
  pass


if __name__ == '__main__':
  t_in = Tensor(np.random.rand(10), name="t_in")
  print("t_in:", t_in)
  perceptron = Linear(t_in.shape([0]), 5)
  t_out = perceptron(t_in)
  print("t_out:", t_out)
  print()
  print("==Backpropagation of t_out==")
  t_out.backward()

  exit(0)
  gt = Tensor(np.random.rand(1, 5), name="ground_truth")
  loss = MSELoss(t_out, gt)
  print("loss:", loss)
  print()
  for p in loss._prev:
    print("[*] ", p)
  loss.backward()
