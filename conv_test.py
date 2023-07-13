#!/usr/bin/env python3
from time import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensor import Tensor


if __name__ == "__main__":
  img = Image.open("./media/cat.jpeg")
  img = img.convert("L")
  img = img.resize((224, 224))
  plt.imshow(img, cmap="gray")
  plt.show()

  img = np.array(img)
  t_in = Tensor(img, name="t_in")
  print("t_in:", t_in)
  gt = Tensor(np.ones((1,1)), name="ground_truth")  # for binary classification

  conv_time = time()
  t_out = t_in.conv2d(3, 6, 5)
  print("t_out:", t_out)
  print("Time elapsed for Conv2D op: %.2f sec"%(time() - conv_time))

  pool_time = time()
  t_out = t_out.maxpool2d()
  print("t_out:", t_out)
  print("Time elapsed for MaxPool2D op: %.2f sec"%(time() - pool_time))

  # TODO: display images in grid
  out_img = Image.fromarray(t_out.data)
  plt.imshow(out_img, cmap="gray")
  plt.show()
