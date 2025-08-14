#!/usr/bin/env python3
import numpy as np

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from picograd.tensor import Tensor
from picograd.util import is_cuda_available
from picograd.backend.device import Devices, Device
from picograd.backend.cuda.utils import *
from picograd.draw_utils import draw_dot
from picograd.loss import CrossEntropyLoss

device = Device(Devices.CPU)
# device = Device(Devices.CUDA) if is_cuda_available() else Device(Devices.CPU)
print("[*] Using device", device.name, "\n")

t1 = np.random.randn(1, 3, 10, 10).astype(np.float32)
a = Tensor(t1, name="a", device=device)
torch_a = torch.tensor(t1)

res = a.maxpool2d()
res.backward()
torch_res = nn.MaxPool2d(kernel_size=2, stride=1)(torch_a).numpy()
assert np.allclose(res.data, torch_res), "MaxPool2d output does not match PyTorch output"
print("[+] MaxPool2D OK")

res = a.avgpool2d()
res.backward()
torch_res = nn.AvgPool2d(kernel_size=2, stride=1)(torch_a).numpy()
assert np.allclose(res.data, torch_res), "AvgPool2d output does not match PyTorch output"
print("[+] AvgPool2D OK")
