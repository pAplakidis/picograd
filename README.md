# PICOGRAD
A from-scratch toy implementation of neural networks, backpropagation, etc

### Note that this library is a work in progress, therefore some features and ops might cause errors or have not been implemented yet.

## Setup

- Build C++ libraries (deprecated, not required)

```
Release:
./build.sh

or Debug:
./build.sh debug
```

## Requirements

- NVIDIA drivers and CUDA toolkit (if using NVIDIA GPU, tested on Linux 6.14.0-27-generic #27~24.04.1-Ubuntu with cuda 12.9)
- Python dependencies

```
python3 -m pip install -r requirements.txt
```

- Other dependencies (Linux)

```
sudo apt-get install graphviz
```

## Give it a try

Examples

```bash
./examples/MNIST_test.py
./examples/MNIST_simple.py
```

Tests

```bash
./test/ops_test.py
```

NOTE: cuda might contain bugs that cause segmentation faults. To help reduce the change of that, run:
```bash
DEBUG=3 PSEUDO_DEBUG=1 ./test/test_dot.py
```

Code

```python
from picograd import Tensor
from picograd.draw_utils import draw_dot

a = Tensor(np.random.randn(100, 50), device=device)
b = Tensor(np.random.randn(50, 100)).to(device)
c = a.dot(b)
d = Tensor(np.random.randn(100, 100), device=device)
e = c + d
e.backward()
draw_dot(e, path="graphs/test")
```

## Debug Levels

1. Print latency and GFLOPS
2. Print device operations
3. Print intermediate representation of kernel (if using device != cpu)
4. Print streaming assembler code that runs on the device (if using device != cpu)

You can set debug levels by assigning the debug value to DEBUG env variable.

## TODO

- Implement CUDA activation functions (and other unary ops)
- Implement CUDA pooling
- Implement BatchNorm1D and 2D (+CUDA)
- debug & optimize CUDA and memory leaks (device data should not be moved to host in ops)
- cudaMallocManaged

- Test on actual neural networks, efficientnet, etc (full training and evaluation of simple models)
- Unit tests

## BUGS

- CUDA MNIST not learning

### DONE

- cuda conv-net
- ops.py + function.py
- conv2d, maxpool, etc
- save/load models - state dict
- good unit tests
- Support CUDA/GPU
- Low Level Debugging: calculate and print FLOPS
- Adam
- debug MNIST
- ops should be MUL, ADD, etc, instead of Linear
- better backward: debug and use deepwalk (prev: only tensors used in current op => recursively call backward())
- Use nn.Module instead of Tensors (manually)
- !!! Support batches !!!
- separate layers => whole model/Module
- Print out a visual graph in order to debug better (can be better)
- Implement convolution (Conv2D backward) (fix padding + more tests)
- Implement maxpool and avgpool (optimize/refactor code) (needs fixing + tests)
- Fix backward pass/gradient decent
- Implement optimization
- Tidy up code and use the operation wrappers for Tensor
- Fully train a toy Net with only Linear layers

## Backlog:

- GEMM with tensorcores
- OpenCL ops
