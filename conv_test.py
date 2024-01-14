#!/usr/bin/env python3
from time import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import nn
from tensor import Tensor
from loss import BCELoss
from optim import *

classes = {0: "not-cat", 1: "cat"}

def train(t_in, gt):
  lr = 1e-3
  epochs = 10
  for i in range(epochs):
    print("[+] epoch", i+1)
    # t1 = t_in.conv2d(3, 6, 3, padding=2, debug=True)
    conv1 = nn.Conv2d(3, 6, 3)
    t_in.layer = conv1
    t1 = conv1(t_in)

    pooling = nn.MaxPool2D()
    # pooling = nn.AvgPool2D()
    t1.layer = pooling
    t2 = pooling(t1)

    t3 = t2.relu()
#
    t4 = t3.flatten()
    fc = nn.Linear(t4.data.shape[0], 1)
    t4.layer = fc
    t5 = fc(t4)

    t6 = t5.softmax()

    print("Pred:", t6.data)
    loss = BCELoss(t6, gt) # for binary classification
    print("Loss:", loss.data)
    loss.backward()

    params = list(loss._prev.copy())
    params.insert(0, loss)
    params = list(reversed(params))

    params = manual_update(params, lr)
    params = reset_grad(params)

  print("\nNetwork Graph:")
  loss.print_graph()

def manual_test(t_in, gt):
  conv_time = time()
  # t1 = t_in.conv2d(3, 6, 3, padding=2, debug=True)
  t1 = t_in.conv2d(3, 6, 3)
  print("t1:", t1)
  print("Time elapsed for Conv2D op: %.2f sec"%(time() - conv_time))
  conv_img = t1.data

  # pool_time = time()
  # t2 = t2.maxpool2d()
  # print("t2:", t2)
  # print("Time elapsed for MaxPool2D op: %.2f sec"%(time() - pool_time))

  # TODO: add ReLU

  t1.flatten()
  fc = nn.Linear(t1.data.shape[0], 1)
  t1.layer = fc
  t2 = fc(t1)
  print("t2:", t2)
  print(t2.data)

  t3 = t2.softmax()
  print("t3:", t3)
  print(t3.data)

  loss = BCELoss(t3, gt) # for binary classification
  print("Loss:", loss.data)
  loss.backward()

  # TODO: display images/feature-maps in grid
  # out_img = Image.fromarray(conv_img)
  # plt.imshow(out_img, cmap="gray")
  # plt.show()


if __name__ == "__main__":
  img = Image.open("./media/cat.jpeg")
  # img = img.convert("L")
  img = img.convert("RGB")
  img = img.resize((112, 112))
  plt.imshow(img, cmap="gray")
  plt.show()

  img = np.array(img)
  img = np.moveaxis(img, -1, 0)
  print(img.shape)
  t_in = Tensor(img, name="t_in")
  # t_in = Tensor(np.random.randint(0, 255, (3, 5, 5)), name="t_in") # NOTE: mock-tensor
  print("t_in:", t_in)
  gt = Tensor(np.ones((1,1)), name="ground_truth")  # for binary classification
  print("Ground Truth:", gt, "=>", classes[gt.data[0][0]])

  train(t_in, gt)
