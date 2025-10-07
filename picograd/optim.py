import numpy as np
from picograd.tensor import Tensor

class Optim:
  def __init__(self, params, lr=0.001):
    self.params = [p for p in params if isinstance(p, Tensor)]
    self.lr = lr

  # not zero_grad since we reset it to ones
  def zero_grad(self):
    for p in self.params:
      if hasattr(p, 'grad') and p.grad is not None:
        p.grad = np.zeros_like(p.grad)
    return self

  def clip_grad_norm_(self, max_norm):
    # Global norm clip: compute total norm and scale if needed
    total_norm_sq = 0.0
    for p in self.params:
      if getattr(p, 'grad', None) is not None:
        total_norm_sq += float((p.grad**2).sum())
    total_norm = np.sqrt(total_norm_sq)
    if total_norm > max_norm:
      scale = max_norm / (total_norm + 1e-6)
      for p in self.params:
        if getattr(p, 'grad', None) is not None:
          p.grad *= scale


class SGD(Optim):
  def step(self):
    for p in self.params:
      if getattr(p, 'grad', None) is None:
        continue
    p.data -= self.lr * p.grad

class Adam(Optim):
  def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    super().__init__(params, lr)
    self.b1 = b1
    self.b2 = b2
    self.eps = eps

    self.state = {}
    self.t = 0

    for p in self.params:
      self.state[id(p)] = {
        'm': np.zeros_like(p.data),
        'v': np.zeros_like(p.data)
      }

  def step(self):
    self.t += 1
    max_grad_norm = 1.0
    self.clip_grad_norm_(max_grad_norm)

    for p in self.params:
      g = getattr(p, 'grad', None)
      if g is None:
        continue
      st = self.state[id(p)]
      m = st['m']
      v = st['v']

      # update biased first and second moment estimates
      m[:] = self.b1 * m + (1.0 - self.b1) * g
      v[:] = self.b2 * v + (1.0 - self.b2) * (g * g)

      # bias correction
      m_hat = m / (1.0 - (self.b1 ** self.t))
      v_hat = v / (1.0 - (self.b2 ** self.t))

      # update parameters (in-place)
      p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
