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
  def __init__(self, params, lr=0.001):
    self.params = params
    self.lr = lr

  # not zero_grad since we reset it to ones
  def zero_grad(self):
    for i in range(len(self.params)):
      self.params[i].t_in.grad = np.zeros_like(self.params[i].t_in.grad)
      self.params[i].t_out.grad = np.zeros_like(self.params[i].t_out.grad)
    return self.params

  def reset_grad(self):
    for i in range(len(self.params)):
      self.params[i].t_in.grad = np.ones_like(self.params[i].t_in.grad)
      self.params[i].t_out.grad = np.ones_like(self.params[i].t_out.grad)
    return self.params


class SGD(Optim):
  def step(self):
    for i in range(len(self.params)):
      self.params[i].weight.data -= self.lr * self.params[i].weight.grad
      self.params[i].bias.data -= self.lr * self.params[i].bias.grad

class Adam(Optim):
  def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    super().__init__(params, lr)
    self.b1 = b1
    self.b2 = b2
    self.eps = eps

    self.mt_1_weight = [None] * len(self.params)
    self.ut_1_weight = [None] * len(self.params)
    self.mt_1_bias = [None] * len(self.params)
    self.ut_1_bias = [None] * len(self.params)
    for i in range(len(self.params)):
      self.mt_1_weight[i] = np.zeros_like(self.params[i].weight.grad)
      self.ut_1_weight[i] = np.zeros_like(self.params[i].weight.grad)

      self.mt_1_bias[i] = np.zeros_like(self.params[i].bias.grad)
      self.ut_1_bias[i] = np.zeros_like(self.params[i].bias.grad)

  def step(self):
    for i in range(len(self.params)):
      # weight
      mt_weight = self.b1 * self.mt_1_weight[i] + (1 - self.b1) * self.params[i].weight.grad
      ut_weight = self.b2 * self.ut_1_weight[i] + (1 - self.b2) * self.params[i].weight.grad**2
      mt_hat_w = mt_weight / (1 - self.b1)
      ut_hat_w = ut_weight / (1 - self.b2)
      self.params[i].weight.data -= self.lr * mt_hat_w / (np.sqrt(ut_hat_w) + self.eps)

      self.mt_1_weight[i] = mt_weight
      self.ut_1_weight[i] = ut_weight

      # bias
      mt_bias = self.b1 * self.mt_1_bias[i] + (1 - self.b1) * self.params[i].bias.grad
      ut_bias = self.b2 * self.ut_1_bias[i] + (1 - self.b2) * self.params[i].bias.grad**2
      mt_hat_b = mt_bias / (1 - self.b1)
      ut_hat_b = ut_bias / (1 - self.b2)
      self.params[i].bias.data -= self.lr * mt_hat_b / (np.sqrt(ut_hat_b) + self.eps)

      self.mt_1_bias[i] = mt_bias
      self.ut_1_bias[i] = ut_bias
