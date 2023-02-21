#!/usr/bin/env python3
import numpy as np
from graphviz import Digraph

OPS = {"Linear": 0,
       "Conv2d": 1,
       "ReLU": 2,
       "Softmax": 3,
       "Sigmoid": 4}

def get_key_from_value(d, val):
  return [k for k, v in d.items() if v == val]

# TODO: implement graph visualization
def trace():
  pass

def graph():
  pass

def draw_dot():
  pass



class Tensor:
  def __init__(self, data: np.array, name="t", _children=(), verbose=False):
    self.name = name
    self.data = data
    self.verbose = verbose
    self._prev = set(_children)
    self.grad = np.ones(self.data.shape)
    self.out = None
    self._ctx = None
    self.prev_op = None
    self._backward = lambda: None

  def __repr__(self):
    if self.verbose:
      return f"Tensor(name={self.name}, shape={str(self.shape())}, data={str(self.data)}, grad={self.grad}), prev_op={get_key_from_value(OPS, self.prev_op)}, prev_tensors={len(self._prev)}"
    else:
      return f"Tensor(name={self.name}, shape={str(self.shape())}, prev_op={get_key_from_value(OPS, self.prev_op)}, prev_tensors={len(self._prev)}"

  def __add__(self, other):
    #children = self._prev.copy()
    #children.add(self)
    #return Tensor(self.data + other.data, _children=children)
    return Tensor(self.data + other.data)

  def __sub__(self, other):
    #children = self._prev.copy()
    #children.add(self)
    #return Tensor(self.data - other.data, _children=children)
    return Tensor(self.data - other.data)

  def __mul__(self, other):
    #children = self._prev.copy()
    #children.add(self)
    #return Tensor(self.data * other.data, _children=children)
    return Tensor(self.data * other.data)

  def __pow__(self, other):
    #children = self._prev.copy()
    #children.add(self)
    #return Tensor(self.data ** other, _children=children)
    return Tensor(self.data ** other)

  def __div__(self, other):
    #children = self._prev.copy()
    #children.add(self)
    #return Tensor(self * (other ** -1), _children=children)
    return Tensor(self * (other ** -1))

  def dot(self, other):
    #children = self._prev.copy()
    #children.add(self)
    #return Tensor(np.dot(self.data, other.data), _children=children)
    return Tensor(np.dot(self.data, other.data))

  def T(self):
    #children = self._prev.copy()
    #children.add(self)
    #return Tensor(self.data.T, _children=children)
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

  # pretty print the graph for this tensor backwards
  def print_graph(self):
    print("[==]", self)
    print("[data]", self.data)
    print("[grad]", self.grad)
    for t0 in reversed(list(self._prev)):
      print("[==]", t0)
      print("[data]", t0.data)
      print("[grad]", t0.grad)
      print()

  def linear(self, w, b):
    self.w = w
    self.b = b
    self.out = self.dot(self.w.data) + self.b.data
    self.out.name = "linearout"
    # TODO: decide whether to add backprop in Layers or Ops
    self.out._prev = self._prev.copy()
    self.out._prev.add(self)
    self.out.prev_op = OPS["Linear"]

    def _backward():
      #print(self.data.shape, self.out.grad.shape)
      if len(self.data.shape) == 1:
        self.w.grad = np.dot(self.data[np.newaxis].T, self.out.grad)
      else:
        self.w.grad = np.dot(self.data.T, self.out.grad)

      #print(self.out.grad.shape)
      self.b.grad = np.sum(self.out.grad, axis=0, keepdims=True)

      #print(self.out.grad.shape, self.w.data.shape)
      self.grad = np.dot(self.out.grad, self.w.data.T)
    self.out._backward = _backward

    return self.out

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

  # TODO: maybe implement a backward for each type of op instead of layer??
  def deep_walk(self):
    def walk(node, visited, nodes):
      if node._ctx:
        [walk(i, visited, nodes) for i in node._ctx.parents if i not in visited]
        nodes.append(node)
      return nodes
    return walk(self, set(), [])

  def backward(self):
    #self.grad = np.ones(self.data.shape)
    print("\n[+] Before backpropagation")
    self.print_graph()
    self._backward()
    for t0 in reversed(list(self._prev)):
      t0._backward()
    print("\n[+] After backpropagation")
    self.print_graph()

  def ReLU(self):
    self.out = Tensor(np.maximum(self.data, np.zeros(self.data.shape)), self._prev.copy())
    self.out._prev.add(self)

    def _backward():
      self.grad += self.out.grad * (self.out.data > 0)
    self.out._backward = _backward

    return self.out

  def tahn(self):
    pass

  def sigmoid(self):
    pass

  def softmax(self):
    exp_val = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
    probs = exp_val / np.sum(exp_val, axis=1, keepdims=True)
    self.out = Tensor(probs, _children=self._prev.copy())
    self.out._prev.add(self)

    def _backward():
      pass
    self.out._backward = _backward()
    return self.out


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
