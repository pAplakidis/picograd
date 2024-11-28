import numpy as np
from picograd.tensor import Tensor

def manual_update(params, lr):
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

class Optim:
  def __init__(self, params, lr=1.0):
    self.params = params
    self.lr = lr

  # not zero_grad since we reset it to ones
  def reset_grad(self):
    for i in range(len(self.params)):
      self.params[i].t_in.grad = np.ones_like(self.params[i].t_in.grad)
      self.params[i].t_out.grad = np.ones_like(self.params[i].t_out.grad)
    return self.params


class SGD(Optim):
  def step(self):
    for i in range(len(self.params)):
      #self.params[i].weight += -self.lr * self.params[i].weight.grad
      self.params[i].weight += self.lr * self.params[i].weight.grad
      #self.params[i].bias += -self.lr * self.params[i].bias.grad
      self.params[i].bias += self.lr * self.params[i].bias.grad

# TODO: implement Adam
class Adam(Optim):
  def step(self):
    pass
