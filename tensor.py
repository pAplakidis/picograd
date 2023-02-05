#!/usr/bin/env python3
import numpy as np

OPS = {"Linear": 0,
       "Conv2d": 1,
       "ReLU": 2,
       "Softmax": 3,
       "Sigmoid": 4}


class Tensor:
  def __init__(self, data: np.array, __children=()):
    self.data = data
    self.grad = np.zeros(len(self.data))
    self.__prev = set(__children)
    self._ctx = None
    self.prev_op = None
    self._backward = lambda: None

  # TODO: write op wrappers here
  def __repr__(self):
    return f"Tensor(shape={str(self.shape())}, data={str(self.data)}, grad={self.grad}), prev_op={str(self.prev_op)}, prev_tensors={len(self.__prev)}"

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
    out = Tensor(np.dot(self.data, w.data) + b.data, self.__prev)
    out.__prev.add(self)
    out.prev_op = OPS["Linear"]

    def _backward():
      # TODO: the shapes are all wrong!
      w.grad = np.dot(self.data.T, self.grad)
      b.grad = np.sum(self.grad, axis=0, keepdims=True)
      self.grad = np.dot(self.grad, w.data.T)
    out._backward = _backward

    return out

  def conv2d(self):
    pass

  # TODO: implement a backward for each type of op
  def deep_walk(self):
    def walk(node, visited, nodes):
      if node._ctx:
        [walk(i, visited, nodes) for i in node._ctx.parents if i not in visited]
        nodes.append(node)
      return nodes
    return walk(self, set(), [])

  # NOTE: here is the code that just calls backward(*args) for each child
  """
  def backward(self):
    # TODO: self.grad = w * next.grad
    # TODO: calc grad for every node
    self.grad = Tensor(np.ones(self.data.shape))

    for t0 in reversed(self.deepwalk()):
      assert (t0.grad is not None)
      grads = t0._ctx.backward(t0.grad)
      grads = [Tensor(g) if g is not None else None for g in ([grads] if len(t0._ctx.parents)== 1 else grads)]

      for t, g in zip(t0._ctx.parents, grads):
        if g is not None:
          assert g.shape == t.shape
          t.grad = g if t.grad is None else (t.grad + g)
      del t0._ctx
  """

  # TODO: implement activation functions here
  def ReLU(self):
    out = Tensor(np.maximum(self.data, np.zeros(self.data.shape)), self.__prev)

    def _backward():
      self.grad += out.grad * (out.data > 0)
    out._backward = _backward

    return out

  def tahn(self):
    pass

  def sigmoid(self):
    pass

  def softmax(self):
    pass


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
