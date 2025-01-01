# A from-scratch toy implementation of neural networks, backpropagation, etc

## Setup

- Build C++ libraries

```
Release:
./build.sh

or Debug:
./build.sh debug
```

- Install python dependencies

```
python3 -m pip install -r requirements.txt
```

## Give it a try

```
./examples/MNIST_test.py
```

## TODO

- save/load models - state dict
- conv2d, maxpool, etc
- good unit tests
- Implement BatchNorm1d and 2d
- Support CUDA for numpy using CuPy => add device property
- Low Level Debugging: calculate and print FLOPS
- Test on actual neural networks (full training and evaluation of simple models)
- Release (make the project cleaner, more robust and usable)

### DONE

- debug MNIST (DONE)
- ops should be MUL, ADD, etc, instead of Linear (DONE)
- better backward: debug and use deepwalk (prev: only tensors used in current op => recursively call backward()) (DONE)
- Use nn.Module instead of Tensors (manually) (DONE)
- !!! Support batches !!! (DONE)
- separate layers => whole model/Module (DONE)
- Print out a visual graph in order to debug better (can be better)
- Implement convolution (Conv2D backward) (fix padding + more tests)
- Implement maxpool and avgpool (optimize/refactor code) (needs fixing + tests)
- Fix backward pass/gradient decent (DONE)
- Implement optimization (DONE)
- Tidy up code and use the operation wrappers for Tensor (DONE)
- Fully train a toy Net with only Linear layers (DONE)

## Backlog:

### Custom backend:

- C using ctypes
- custom cuda ops
- cuda driver
- GPU driver
