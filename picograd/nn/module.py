from enum import Enum, auto
from picograd.tensor import Tensor
from picograd.backend.device import Devices, Device


class LayerType(Enum):
  NOLAYER = auto()
  LINEAR = auto()
  CONV2D = auto()

  MAXPOOL2D = auto()
  AVGPOOL2D = auto()

  BATCHNORM2D = auto()
  LAYERNORM = auto()

  RNN = auto()
  LSTM = auto()

  def __str__(self):
    return self.name


class Layer:
  def __init__(self, device = Device(Devices.CPU)):
    self.type = None
    self.t_in = None
    self.t_out = None
    self.train = True
    self.device = device
    self._params: dict[str, Tensor] = {}

    self.subgraph_name = None
    self._subgraph_nodes = []

  def register_param(self, name: str, tensor: Tensor):
    tensor.name = name
    self._params[name] = tensor
    return tensor

  def parameters(self):
    """Returns a list of Tensors registered on this layer."""
    return list(self._params.values())

  def to(self, device: Device):
    self.device = device
    for param in self.parameters(): param.to(device)
    return self

  def _track(self, t: Tensor):
    """Tracks tensors that belong to this layer's subgraph for better visualization"""
    self._subgraph_nodes.append(t)
    t.layer = self
    return t


# TODO: Sequential
class Module:
  def __init__(self):
    self.params = []
    self.train = True
    self.device = Device(Devices.CPU)

  def to(self, device: Device):
    self.device = device
    for param in self.get_params(): param.to(device)
    return self

  def train_mode(self):
    self.train = True
    for param in self.get_params(): param.train = True
    return self

  def eval_mode(self):
    self.train = False
    for param in self.get_params(): param.train = False
    return self

  def forward(self):
    return None

  def __call__(self, *params):
    return self.forward(*params)

  def get_layers(self):
    layers = []
    for _, v in self.__dict__.items():
      if isinstance(v, Layer):
        layers.append(v)
    return layers

  def get_params(self):
    params = []
    for layer in self.get_layers():
      params.extend(layer.parameters())
    return params


class Sequential(Module):
  def __init__(self, *layers: Layer):
    super().__init__()
    self.layers = layers

  def forward(self, x: Tensor) -> Tensor:
    for layer in self.layers:
      x = layer(x)
    return x

class ModuleList(Module):
  def __init__(self, modules: list[Module] = []):
    super().__init__()
    self.modules = modules

  def append(self, module: Module):
    self.modules.append(module)

  def __getitem__(self, index: int) -> Module:
    return self.modules[index]

  def forward(self, *args, **kwargs):
    outputs = []
    for module in self.modules:
      outputs.append(module(*args, **kwargs))
    return outputs
