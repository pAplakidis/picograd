import picograd.nn as nn
from picograd import Tensor
from .self_attention import MultiHeadAttention
from .layernorm import LayerNorm


# FIXME: implement missing
class FeedForward(nn.Module):
  """ a simple feed-forward neural network """

  def __init__(self, n_embed: int, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, 4 * n_embed),
      nn.ReLU(),
      nn.Linear(4 * n_embed, n_embed),
      nn.Dropout(dropout),
    )

  def __call__(self, x):
    return self.net(x)

class TransformerBlock(nn.Module):
  """ Transformer block: communication followed by computation """

  def __init__(self, n_embed: int, n_head: int, block_size, dropout=0.1):
    # n_embed: embedding dimension, n_head: the number of heads we'd like
    super().__init__()
    head_size = n_embed // n_head
    self.sa = MultiHeadAttention(n_head, block_size, head_size, n_embed, dropout)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))  # unlike paper, we do pre-norm instead of post-norm (better results)
    x = x + self.ffwd(self.ln2(x))
    return x
