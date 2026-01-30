import numpy as np

import picograd.nn as nn
from picograd import Tensor


class Head(nn.Module):
  """ one head of self-attention"""

  def __init__(self, head_size: int, block_size: int, n_embed: int, dropout=0.1):
    super().__init__()
    self.key   = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    # register buffer: moved to device + saved in state_dict, but not a parameter (no grad)
    # tril: lower triangular matrix
    # self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    self.tril = Tensor(np.tril(np.ones((block_size, block_size), dtype=np.float32)))
    self.dropout = nn.Dropout(dropout)

  def __call__(self, x: Tensor):
    B, T, C = x.shape
    k = self.key(x)     # (B,T,head_size)
    q = self.query(x)   # (B,T,head_size)

    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
    # NOTE: this is for decoder self-attention, so we need to mask out the future tokens
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
    wei = wei.softmax(axis=-1)  # (B, T, T)
    wei = self.dropout(wei)
    
    # perform the weighted aggregation of the values
    v = self.value(x)   # (B,T,C)
    out = wei @ v       # (B, T, T) @ (B, T, C) -> (B, T, C)
    return out


class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel """

  def __init__(self, num_heads: int, block_size: int, head_size: int, n_embed: int, dropout=0.1):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size, block_size, n_embed, dropout=dropout) for _ in range(num_heads)])
    self.proj = nn.Linear(num_heads * head_size, n_embed)
    self.dropout = nn.Dropout(dropout)
  
  def __call__(self, x: Tensor):
    out = Tensor.cat([h(x) for h in self.heads], axis=-1)
    out = self.dropout(self.proj(out))
    return out
