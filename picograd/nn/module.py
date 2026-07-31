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

  @property
  def parameters(self):
    """Returns a list of Tensors registered on this layer."""
    return list(self._params.values())

  def to(self, device: Device):
    self.device = device
    for param in self.parameters: param.to(device)
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

    self._buffers = {}
    self._modules = {}

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

  @property
  def layers(self):
    return {name: v for name, v in self.__dict__.items() if isinstance(v, Layer)}

  def get_params(self):
    params = []
    for layer in self.layers.values():
      params.extend(layer.parameters)
    return params

  def state_dict(self):
    state = {}
    for name, layer in self.layers.items():
      for param_name, param in layer._params.items():
        param.realize()
        state[f"{name}.{param_name}"] = param
    return state

  def load_state_dict(self, state_dict: dict):
    for name, layer in self.layers.items():
      for param_name, param in layer._params.items():
        key = f"{name}.{param_name}"
        if key in state_dict:
          param.data = state_dict[key].data
        else:
          raise KeyError(f"Parameter {key} not found in state_dict")


class Sequential(Module):
  def __init__(self, *layers: Layer):
    super().__init__()
    self._layers = layers

  def forward(self, x: Tensor) -> Tensor:
    for layer in self._layers:
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
