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
./examples/MNIST.py
```

Tests

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

Code

```python
from picograd import Tensor
from picograd.draw_utils import draw_dot

a = Tensor.random((100, 50))
b = Tensor.random((50, 100))
c = a.dot(b)
d = Tensor.random((100, 100))
e = c + d
e.backward()
draw_dot(e, path="graphs/test")
```

## Debug Levels

1. Print kernel summary
2. Render kernel code
3. Log device internals
4. Print intermediate representation of kernel (if using device != cpu)
5. Print streaming assembler code that runs on the device (if using device != cpu)

You can set debug levels by assigning the debug value to DEBUG env variable.

```bash
DEBUG=3 ./test/test_ops.py
```

## TODO

- make gradients a Tensor so that backwards can be used with lazy
- save/load models - state dict
- rewrite device to use Buffer class
- shapetracker + conv2d padding
- full metal support (20 ops)

- replace ctypes with pycuda
- CUDA activation functions (and other unary ops)
- CUDA pooling
- BatchNorm1D & 2D, LayerNorm (CUDA)
- Attention, self-attention, transformer (CUDA)

- GRU
- EfficientNet classifier
- Stable Diffusion
- Better AST => better Lazy Buffers => ScheduleItems
- Generic Renderer using pattern matcher
- allocate memory on realize + UOps.LOAD only (+ store on UOps.STORE only)
- JIT
- kernel fusion
- LLAMA (?)

## BUGS

- MNIST_simple (cuda) - illegal address on relu out.grad read + out of memory after some iterations

### DONE

- Conv2D trick
- Lazy buffers, scheduler, linearizer
- RNN, LSTM,
- Unit tests
- Residual connections
- CrossEntropyLoss CUDA
- debug & optimize CUDA and memory leaks (device data should not be moved to host in ops)
- CUDA sometimes segfaults for relu and softmax kernels
- cuda conv-net
- ops.py + function.py
- conv2d, maxpool, etc
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
- cudaMallocManaged
- Test on actual neural networks, efficientnet, etc (full training and evaluation of simple models)
- userspace driver (CLDevice, hook ioctl, etc)
