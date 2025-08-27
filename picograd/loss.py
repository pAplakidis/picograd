#!/usr/bin/env python3
import numpy as np
from picograd.tensor import *
from picograd.backend.function import *

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
def MSELoss(z: Tensor, y: Tensor) -> Tensor:
  assert z.shape == y.shape, f"Z Tensor doesn't have the same shape as ground-truth Y: z.shape={str(z.shape)}, y.shape={str(y.shape)}"
  loss_val = 1/z.shape[0] * np.sum((z.data-y.data) ** 2)
  t = Tensor(loss_val, name="mseloss_out",  _prev=(z,))
  t.prev_op = OPS.MSELoss
  t.grad = 2 * (z.data - y.data) / y.shape[0]
  return t

# Mean Absolute Error Loss
def MAELoss(z: Tensor, y: Tensor) -> Tensor:
  assert (n := z.shape[0]) == y.shape[0], f"Z Tensor doesn't have the same shape as ground-truth Y: z.shape={str(z.shape)}, y.shape={str(y.shape)}"
  loss_val = 1/n * np.sum(np.abs(z.data-y.data))
  t = Tensor(loss_val, name="maeloss_out", _prev=(z,))
  t.prev_op = OPS.MAELoss
  return t

# TODO: always outputs 0?? (might be just the tests, but double check!)
# Binary Cross Entropy Loss
def BCELoss(z: Tensor, y: Tensor) -> Tensor:
  # FIXME: assert all dimensions
  assert (n := z.shape[0]) == y.shape[0], f"Z Tensor doesn't have the same shape as ground-truth Y: z.shape={str(z.shape)}, y.shape={str(y.shape)}"
  samples = z.shape[0]
  y_pred_clipped = np.clip(z.data, 1e-7, 1 - 1e-7)

  term_0 = (1 - y.data) * np.log(1 - y_pred_clipped + 1e-7)
  term_1 = y.data * np.log(y_pred_clipped + 1e-7)
  loss_val = -np.mean(term_0+term_1, axis=0)
  t = Tensor(loss_val, name="bceloss_out",  _prev=(z,))
  t.prev_op = OPS.BCELoss
  t.grad = (z.data - y.data) / (z.data * (1 - y.data))
  return t

# Categorical Cross Entropy
def CrossEntropyLoss(z: Tensor, y: Tensor) -> Tensor:
  assert len(z.shape) == 2, "Z Tensor must be 2D (batch_size, num_classes)"
  assert len(y.shape) == 1, "Ground-truth Y must be 1D (batch_size,)"
  assert z.shape[0] == y.shape[0], "Z Tensor and ground-truth Y must have the same batch size"

  func = CrossEntropy(z.device.name)
  if z.device.name == Devices.CPU:
    out = Tensor(
      func.forward(z, y),
      name="crossentropyloss_out",
      _prev=(z,),
      requires_grad=False
    )
  else:
    out = Tensor(
      device_data=func.forward(z, y),
      shape=(z.shape[0],),
      _prev=(z,),
      requires_grad=False,
      device=z.device
    )
  out.prev_op = OPS.CrossEntropyLoss
  out._backward = lambda: func.backward()
  return out

# Negative Log Likelihood Loss
def NLLLoss(z: Tensor, y: Tensor, sigma=1.0):
  dist = np.random.normal(z.data, sigma)
  return None


if __name__ == '__main__':
  t1 = Tensor(np.random.rand(3))
  t2 = Tensor(np.random.rand(3))
  print(MSELoss(t1, t2))
