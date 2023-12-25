#!/usr/bin/env python3
from time import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import nn
from tensor import Tensor
from loss import BCELoss

classes = {0: "not-cat", 1: "cat"}

if __name__ == "__main__":
  img = Image.open("./media/cat.jpeg")
  # img = img.convert("L")
  img = img.convert("RGB")
  img = img.resize((224, 224))
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

  conv_time = time()
  t1 = t_in.conv2d(3, 6, 3)
  print("t1:", t1)
  print("Time elapsed for Conv2D op: %.2f sec"%(time() - conv_time))
  conv_img = t1.data
  exit(0)

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
