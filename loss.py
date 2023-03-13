#!/usr/bin/env python3
import numpy as np
from tensor import *

# TODO: all losses need to be classes that inherit from this abstract one
"""
class Loss:
  def __init__(self):
    pass

class MSELoss(Loss):
  def __call__(self):
    # code that calculcates the loss value
"""

# z: network output, y: ground truth
# Mean Squared Error Loss
def MSELoss(z: Tensor, y: Tensor):
  assert (n := z.shape()[0]) == y.shape()[0], f"Z Tensor doesn't have the same shape as ground-truth Y: z.shape={str(z.data.shape)}, y.shape={str(y.data.shape)}"
  loss_val = 1/n * np.sum((z.data-y.data) ** 2)
  t = Tensor(loss_val, name="mseloss_out", _children=z._prev.copy())
  t._prev.append(z)
  t.prev_op = OPS["MSELoss"]
  return t

# Mean Absolute Error Loss
def MAELoss(z: Tensor, y: Tensor):
  assert (n := z.shape()[0]) == y.shape()[0], f"Z Tensor doesn't have the same shape as ground-truth Y: z.shape={str(z.data.shape)}, y.shape={str(y.data.shape)}"
  loss_val = 1/n * np.sum(np.abs(z.data-y.data))
  t = Tensor(loss_val, name="maeloss_out", _children=z._prev.copy())
  t._prev.append(z)
  t.prev_op = OPS["MAELoss"]
  return t

# Binary Cross Entropy Loss
def BCELoss(z: Tensor, y: Tensor):
  assert (n := z.shape()[0]) == y.shape()[0], f"Z Tensor doesn't have the same shape as ground-truth Y: z.shape={str(z.data.shape)}, y.shape={str(y.data.shape)}"
  samples = z.data.shape[0]
  y_pred_clipped = np.clip(z.data, 1e-7, 1 - 1e-7)

  term_0 = (1 - y.data) * np.log(1 - y_pred_clipped + 1e-7)
  term_1 = y.data * np.log(y_pred_clipped + 1e-7)
  loss_val = -np.mean(term_0+term_1, axis=0)
  t = Tensor(loss_val, name="bceloss_out", _children=z._prev.copy())
  t._prev.append(z)
  t.prev_op = OPS["BCELoss"]
  return t

# Categorical Cross Entropy
def CrossEntropyLoss(z: Tensor, y: Tensor):
  assert (n := z.shape()[0]) == y.shape()[0], f"Z Tensor doesn't have the same shape as ground-truth Y: z.shape={str(z.data.shape)}, y.shape={str(y.data.shape)}"
  samples = z.data.shape[0]
  y_pred_clipped = np.clip(z.data, 1e-7, 1 - 1e-7)

  if len(y.data.shape) == 1:
    correct_confidences = y_pred_clipped[range(samples), y.data]
  elif len(y.data.shape) == 2:
    correct_confidences = np.sum(y_pred_clipped * y.data, axis=1)

  loss_val = -np.log(correct_confidences)
  t = Tensor(loss_val, name="crossentropyloss_out", _children=z._prev.copy())
  t._prev.append(z)
  t.prev_op = OPS["CrossEntropyLoss"]
  return t

# Negative Log Likelihood Loss
def NLLLoss(z: Tensor, y: Tensor, sigma=1.0):
  dist = np.random.normal(z.data, sigma)

  return None


if __name__ == '__main__':
  t1 = Tensor(np.random.rand(3))
  t2 = Tensor(np.random.rand(3))
  print(MSELoss(t1, t2))
