import numpy as np
from picograd import Tensor

class Optim:
  def __init__(self, params, lr=0.001):
    self.params = [p for p in params if isinstance(p, Tensor)]
    self.lr = lr

  # not zero_grad since we reset it to ones
  def zero_grad(self):
    for p in self.params:
      if hasattr(p, "grad") and p.grad is not None:
        p.grad = np.zeros_like(p.grad)
    return self

  def clip_grad_norm_(self, max_norm):
    # Global norm clip: compute total norm and scale if needed
    total_norm_sq = 0.0
    for p in self.params:
      if getattr(p, "grad", None) is not None:
        total_norm_sq += float((p.grad**2).sum())
    total_norm = np.sqrt(total_norm_sq)
    if total_norm > max_norm:
      scale = max_norm / (total_norm + 1e-6)
      for p in self.params:
        if getattr(p, "grad", None) is not None:
          p.grad *= scale


class SGD(Optim):
  def step(self):
    for p in self.params:
      if getattr(p, "grad", None) is None:
        continue
      p.data -= self.lr * p.grad

class Adam(Optim):
  def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
    super().__init__(params, lr)
    self.b1 = b1
    self.b2 = b2
    self.eps = eps
    self.weight_decay = weight_decay

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
      g = getattr(p, "grad", None)
      if g is None:
        continue
      st = self.state[id(p)]
      m = st['m']
      v = st['v']

      # L2 regularization style
      if self.weight_decay:
        g.data = g.data + self.weight_decay * p.data

      # update biased first and second moment estimates
      m[:] = self.b1 * m + (1.0 - self.b1) * g        # Momentum
      v[:] = self.b2 * v + (1.0 - self.b2) * (g * g)  # RMSProp

      # bias correction
      m_hat = m / (1.0 - (self.b1 ** self.t))
      v_hat = v / (1.0 - (self.b2 ** self.t))

      # update parameters (in-place)
      p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

  
class AdamW(Optim):
  def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01):
    super().__init__(params, lr)
    self.b1 = b1
    self.b2 = b2
    self.eps = eps
    self.weight_decay = weight_decay

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
      g = getattr(p, "grad", None)
      if g is None:
        continue
      st = self.state[id(p)]
      m = st['m']
      v = st['v']

      # update biased first and second moment estimates
      m[:] = self.b1 * m + (1.0 - self.b1) * g        # Momentum
      v[:] = self.b2 * v + (1.0 - self.b2) * (g * g)  # RMSProp

      # bias correction
      m_hat = m / (1.0 - (self.b1 ** self.t))
      v_hat = v / (1.0 - (self.b2 ** self.t))

      # update parameters (in-place)
      p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

      # decoupled weight decay
      if self.weight_decay:
        p.data -= self.lr * self.weight_decay * p.data


# Learning Rate Schedulers

class ReduceLROnPlateau:
  def __init__(
    self,
    optmizer: Optim,
    mode="min",
    factor=0.1,
    patience=10,
    threshold=1e-4,
    threshhold_mode="rel",
    cooldown=0,
    min_lr=0.0,
    eps=1e-8,
    verbose=False
  ):
    self.optimizer = optmizer
    self.mode = mode                        # 'min' or 'max'
    self.factor = factor
    self.patience = patience
    self.threshold = threshold
    self.threshhold_mode = threshhold_mode  # 'rel' or 'abs'
    self.cooldown = cooldown
    self.min_lr = min_lr
    self.eps = eps
    self.verbose = verbose

    self.best = None
    self.num_bad_epochs = 0
    self.in_cooldown = False

  def _is_better(self, metric):
    if self.best is None:
      return True
    
    if self.mode == "min":
      if self.threshhold_mode == "rel":
        return metric < self.best * (1 - self.threshold)
      else:
        return metric < self.best - self.threshold
    else:
      if self.threshhold_mode == "rel":
        return metric > self.best * (1 + self.threshold)
      else:
        return metric > self.best + self.threshold

  def _reduce_lr(self):
    old_lr = self.optimizer.lr
    new_lr = max(old_lr * self.factor, self.min_lr)
    if old_lr - new_lr > self.eps:
      self.optimizer.lr = new_lr

  def step(self, metric):
    if self.best is None:
      self.best = metric
      return
    
    if self.in_cooldown:
      self.cooldown_counter -= 1
      if self.cooldown_counter <= 0:
        self.in_cooldown = False

    if self._is_better(metric):
      self.best = metric
      self.num_bad_epochs = 0
    else:
      self.num_bad_epochs += 1

    if (not self.in_cooldown) and (self.num_bad_epochs > self.patience):
      self._reduce_lr()
      if self.verbose:
        print(f"[ReduceLROnPlateau] Reducing learning rate to: {self.optimizer.lr:.6e}")

      self.in_cooldown = True
      self.cooldown_counter = self.cooldown
      self.num_bad_epochs = 0
