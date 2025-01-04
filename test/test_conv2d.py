#!/usr/bin/env python3
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tqdm import tqdm


# TODO: find a way to import without this
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import picograd.nn as nn
from picograd.tensor import Tensor
from picograd.loss import *
from picograd.optim import *
from picograd.draw_utils import draw_dot

BS = 2

def get_data():
  (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
  X_train = X_train / 255.0
  X_test = X_test / 255.0
  return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = get_data()

  weight = Tensor(np.random.randint(0, 255, (2, 1, 3, 3), dtype=np.uint8), "conv2D_kernel")
  bias = Tensor(np.zeros((2, 1, 1, 1)), name="bias")
  X = Tensor(np.expand_dims(X_train[0:BS], axis=1))
  print(X.shape, weight.shape, bias.shape)
  X.conv2d(weight, bias, 1, 1)
