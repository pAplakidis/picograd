import numpy as np
from tensor import Tensor

class SGD:
  def __init__(self, params, lr=1.0):
    self.params = params
    self.lr = lr

  def reset_grad(self):
    for i in range(len(self.params)):
      self.params[i].grad = np.ones_like(self.params[i].grad)
    return self.params

  def step(self):
    for i in range(len(self.params)):
      #self.params[i].weight += -self.lr * self.params[i].weight.grad
      self.params[i].weight += self.lr * self.params[i].weight.grad
      #self.params[i].bias += -self.lr * self.params[i].bias.grad
      self.params[i].bias += self.lr * self.params[i].bias.grad
