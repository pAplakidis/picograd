# A from-scratch toy implementation of neural networks, backpropagation, etc

## Give it a try
```
python simple_test.py
python conv_test.py
```

## TODO:
* Print out a visual graph in order to debug better (DONE, can be better)
* Fix backward pass/gradient decent (DONE)
* Implement optimization  (DONE)
* Tidy up code and use the operation wrappers for Tensor  (DONE)
* Fully train a toy Net with only Linear layers (DONE)
* Implement convolution (Conv2D backward) (DONE, fix padding + more tests)
* Implement maxpool and avgpool (optimize/refactor code) (DONE, needs fixing + tests)
* Use nn.Module instead of Tensors (manually)
* Support CUDA for numpy using CuPy => add device property
* Support batches
* Low Level Debugging: calculate FLOPS

* separate layers => whole model/Module
* save/load models
* Implement BatchNorm1d and 2d
* better backward: debug and use deepwalk
* Test on actual neural networks (full training and evaluation of simple models)
* Release (make the project cleaner, more robust and usable)

