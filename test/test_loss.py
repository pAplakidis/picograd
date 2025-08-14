import torch
import torch.nn as nn
import numpy as np
from picograd.tensor import Tensor
from picograd.loss import CrossEntropyLoss

if __name__ == "__main__":
  softmax = nn.Softmax()
  t1 = torch.tensor([0.4, 0.5, 0.8, 0.3]).float()
  gt = torch.tensor([0, 0, 1, 0]).float()
  t2 = torch.tensor([0.8, 0.8, 0.8, 0.8]).float()

  loss_func = nn.CrossEntropyLoss()

  print("pytorch")
  print(softmax(t1), softmax(t2))
  print(loss_func(t1, gt).item)
  print(loss_func(t2, gt).item)
  print()

  t1 = Tensor(np.array([[0.1, 0.1, 0.7, 0.1]])).softmax()
  gt = Tensor(np.array([[0, 0, 1, 0]]))
  t2 = Tensor(np.array([[0.8, 0.8, 0.8, 0.8]])).softmax()

  print("picograd")
  print(t1.data, t2.data)
  print(CrossEntropyLoss(t1, gt).data[0])
  print(CrossEntropyLoss(t2, gt).data[0])
  print()

