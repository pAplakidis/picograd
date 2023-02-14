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
    self.grad = None
    self.__prev = set(__children)
    self._ctx = None
    self.prev_op = None
    self._backward = lambda: None

  # TODO: write op wrappers here
  def __repr__(self):
    return f"Tensor(shape={str(self.shape())}, data={str(self.data)}, grad={self.grad}), prev_op={str(self.prev_op)}, prev_tensors={len(self.__prev)}"

  def __add__(self, other):
    return Tensor(self.data + other.data)

  def __sub__(self, other):
    return Tensor(self.data - other.data)

  def __mul__(self, other):
    return Tensor(self.data * other.data)

  def __pow__(self, other):
    return Tensor(self.data ** other)

  def __div__(self, other):
    return self * (other ** -1)

  def dot(self, other):
    return Tensor(np.dot(self.data, other.data))

  def T(self):
    return Tensor(self.data.T)

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

  def linear(self, w, b):
    out = self.dot(w.data) + b.data
    out.__prev = self.__prev.copy()
    out.__prev.add(self)
    out.prev_op = OPS["Linear"]

    def _backward():
      self.grad = np.ones_like(w)
      print(self.data.shape, self.grad.shape)
      #print(out.grad.shape, self.grad.shape)
      w.grad = np.dot(self.data.T, self.grad)
      #w.grad = np.dot(out.grad, self.grad.T)

      #print(out.grad.shape)
      b.grad = np.sum(self.grad, axis=0, keepdims=True)
      #b.grad = np.sum(out.grad, axis=1, keepdims=True)

      #print(w.data.shape, out.grad.shape)
      self.grad = np.dot(self.grad, w.data.T)
      #self.grad = np.dot(w.data.T, out.grad)
    out._backward = _backward

    return out

  def conv2d(self):
    pass

  def batchnorm1d(self):
    pass

  def batchnorm2d(self):
    pass

  def maxpool(self):
    pass

  def avgpool(self):
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
  def backward(self):
    """
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
    #self.grad = np.ones_like(self.data) # TODO: this has to be the same shape as the neurons
    self._backward()

  # TODO: implement activation functions here
  def ReLU(self):
    out = Tensor(np.maximum(self.data, np.zeros(self.data.shape)), self.__prev.copy())
    out.__prev.add(self)

    def _backward():
      self.grad += out.grad * (out.data > 0)
    out._backward = _backward

    return out

  def tahn(self):
    pass

  def sigmoid(self):
    pass

  def softmax(self):
    exp_val = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
    probs = exp_val / np.sum(exp_val, axis=1, keepdims=True)
    out = Tensor(probs, __children=self.__prev.copy())
    out.__prev.add(self)

    def _backward():
      pass
    out._backward = _backward()
    return out


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
