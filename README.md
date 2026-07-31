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
- Optimally Python3.12

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

## Lazy

Picograd is not lazy by default (still working on that), but optionally you can add:

```bash
LAZY=1
```

as env variable before running any script and it will run ops lazily using a pseudo-compiler for now.
e.g.:

```bash
LAZY=1 python examples/MNIST_simple.py
```

The goal is for the library to be 100% lazy, with a proper AST builder, scheduler, kernel fusion, etc.

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

## Visualization

Picograd can record a lightweight JSON trace of tensor ops, eager function timings, lazy schedules, generated kernels, kernel timings, copy events, UOps, and backend code artifacts.

Run any script with `VIZ=1`:

```bash
VIZ=1 python3 examples/MNIST_simple.py
```

Then open the static viewer:

```bash
python3 -m picograd.viz.serve
```

Open `http://127.0.0.1:8000/index.html`. The default trace is written to `picograd/viz/traces/latest.json`.

Viewer shows performance event spans with ms labels, clickable kernel source/UOps/assembly tabs, plus search, sort, and zoom controls.

Minimal example:

```python
from picograd import Tensor

a = Tensor([1, 2, 3], requires_grad=False, lazy=False)
b = Tensor([4, 5, 6], requires_grad=False, lazy=False)
c = a + b
print(c.tolist())
```

Run and view it:

```bash
VIZ=1 python3 example.py
python3 -m picograd.viz.serve
```

Quick MNIST-style one-forward smoke script:

```bash
VIZ=1 python3 examples/MNIST_simple_viz.py
python3 -m picograd.viz.serve
```

Remote CUDA MNIST-style smoke example:

On your Mac, keep this SSH tunnel open:

```bash
ssh -N -L 8000:127.0.0.1:8000 user@host
```

On the CUDA host:

```bash
cd ~/Dev/picograd
VIZ=1 CUDA=1 LAZY=1 PICOGRAD_VIZ_PATH=/tmp/picograd-mnist-simple-viz.json python3 examples/MNIST_simple_viz.py
python3 -m picograd.viz.serve --host 127.0.0.1 --port 8000 --trace /tmp/picograd-mnist-simple-viz.json
```

Open `http://localhost:8000/index.html` on your Mac.

For custom trace paths:

```bash
VIZ=1 PICOGRAD_VIZ_PATH=/tmp/picograd-trace.json python3 example.py
python3 -m picograd.viz.serve --trace /tmp/picograd-trace.json
```

If running on a remote CUDA host over SSH, record and serve on the CUDA host:

```bash
VIZ=1 CUDA=1 LAZY=1 python3 example.py
python3 -m picograd.viz.serve --host 127.0.0.1 --port 8000
```

Then on your local machine, forward the port:

```bash
ssh -L 8000:localhost:8000 pavlos@bigg
```

Open `http://localhost:8000/index.html` locally. Use `--host 0.0.0.0` only if you intentionally want the viz server exposed on the remote host network.

## TODO

- save/load models - state dict
- rewrite device to use Buffer class
- shapetracker + conv2d padding
- full metal support (20 ops)
- cleanup renderer and tensor: generic unary ops for activation functions, etc
- allocate memory on realize + UOps.LOAD only (+ store on UOps.STORE only)
- MNIST: lazy CUDA much slower than CPU?

- Better AST => better Lazy Buffers => ScheduleItems
- Generic Renderer using pattern matcher
- JIT
- kernel fusion

- replace ctypes with pycuda (?)
- GRU
- EfficientNet classifier
- Stable Diffusion
- LLAMA (?)

## BUGS

- MNIST_simple (cuda) - illegal address on relu out.grad read + out of memory after some iterations

### DONE

- make gradients a Tensor so that backwards can be used with lazy
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
