#!/usr/bin/env python3
from time import time
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd.tensor import Tensor
from picograd.util import is_cuda_available
from picograd.backend.device import Devices, Device
from picograd.draw_utils import draw_dot
from picograd.loss import CrossEntropyLoss


# device = Device(Devices.CPU)
device = Device(Devices.CUDA) if is_cuda_available() else Device(Devices.CPU)
print("[*] Using device", device.name, "\n")

t1 = np.random.randn(100, 50).astype(np.float32)
t2 = np.random.randn(50, 100).astype(np.float32)
t3 = np.random.randn(100, 100).astype(np.float32)

a = Tensor(t1, name="a", device=device)
b = Tensor(t2, name="b", device=device)
d = Tensor(t3, name="d", device=device)

c = a.dot(b)
c.name = "c"

tmp = c + d
e = tmp.relu()
e.name = "e"
e.backward()

exit(0)
f = Tensor(np.random.rand(100, 100), device=device) # FIXME: test loss
loss = CrossEntropyLoss(d, f)
loss.backward()
