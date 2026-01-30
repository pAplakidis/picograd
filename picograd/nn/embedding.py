import numpy as np
from typing import Any, Optional
from .module import Layer, LayerType
from picograd.tensor import Tensor
from picograd.backend.device import Device, Devices
from picograd.backend.dtypes import dtypes


class Embedding(Layer):
  def __init__(
      self,
      num_embeddings: int,
      embedding_dim: int,
      padding_idx: Optional[int] = None,
      max_norm=None,
      norm_type: float = 2.0,
      scale_grad_by_freq: bool = False,
      sparse: bool = False,
      _weight: Optional[Tensor] = None,
      _freeze: bool = False,
      device: Optional[Device] = Device(Devices.CPU),
      dtype: Optional[dtypes] = None
    ):
    super().__init__()
    self.type = LayerType.LINEAR

    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.padding_idx = padding_idx
    self.max_norm = max_norm
    self.norm_type = norm_type
    self.scale_grad_by_freq = scale_grad_by_freq
    self.sparse = sparse
    self._weight = _weight
    self._freeze = _freeze
    self.device = device
    self.dtype = dtype

    self.weight = Tensor.random((num_embeddings, embedding_dim), device=self.device, name="embedding-weight")
    # TODO: with no_grad()
    if padding_idx is not None: self.weight[padding_idx] = 0.0

  def __call__(self, x: Tensor) -> Tensor:
      emb = self.weight[x.numpy()]
      # if self.padding_idx is not None:
      #    mask = (x == self.padding_idx).unsqueeze(-1)
      #    emb = emb.masked_fill(mask, 0.0) # TODO: implement this
      return emb
