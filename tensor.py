#!/usr/bin/env python3
import numpy as np

class Tensor:
  def __init__(self, data: np.array, __children=()):
    self.data = data
    self.grad = 0.0
    self.__prev = set(__children)

  # TODO: write op wrappers here
  def __repr__(self):
    return f"Tensor(data={self.data}, shape={self.shape})"

  def __add__(self, other):
    return Tensor(self.data + other.data)

  def __mul__(self, other):
    return Tensor(self.data * other.data)

  def item(self):
    return self.data

  def shape(self, idxs=None):
    if idxs is None:
      return self.data.shape
    ret = []
    shp = self.data.shape
    for idx in idxs:
      ret.append(shp[idx])
    
    if len(ret) == 1:
      ret = int(ret[0])
    return ret

  def mean(self):
    return np.mean(self.data)

  # TODO: make operations for neural networks here

  # TODO: the weight shape defines the output shape (bias needs to be the same shape as output)
  def linear(self, w, b):
    #return (self.data * w) + b
    return np.dot(self.data, w.data) + b.data

  # TODO: implement activation functions here


if __name__ == '__main__':
  arr = np.random.rand(3)
  t = Tensor(arr)
  print(t.item())
  print(t.shape())

  w = np.random.rand(3, 4)
  b = np.random.rand(4)
  print(w)
  print(w.shape)
  print(b)
  print(b.shape)
  l = t.linear(w, b)
  print(l)
  print(l.shape)
