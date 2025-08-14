#!/usr/bin/env python3
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd import Tensor
from picograd.backend.device import Devices, Device

device = Device(Devices.CUDA)

a1 = np.array([[1, 2, 3], [4, 5, 6]])
a2 = np.array([[4, 5, 6], [7, 8, 9]])

t1 = Tensor(a1, name="t1", device=device)
print(t1)
t2 = Tensor(a2, name="t2", device=device)
print(t2)

t3 = t1 + t2
print(t3)
t3.backward()

t3 = t1 * t2
print(t3)
t3.backward()

a1 = np.array([[1, 2, 3], [4, 5, 6]])
a2 = np.array([[4, 5], [6, 7], [8, 9]])
t1 = Tensor(a1, name="t1", device=device)
t2 = Tensor(a2, name="t2", device=device)
t3 = t1.dot(t2)
print(t3)
t3.backward()

a = Tensor(np.random.randn(1, 3, 32, 32), device=device)  # (batch_size, channels, height, width)
w = Tensor(np.random.randn(16, 3, 3, 3), device=device)  # (out_channels, kernel_height, kernel_width)
b = Tensor(np.zeros((16,)), device=device)  # (out_channels,)
c = a.conv2d(w, b, 3, 16)
# print(c)
print(c.shape)
c.backward()
print(a.grad, b.grad, c.grad) # FIXME: cuda error (only for conv2d)
