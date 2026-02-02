import numpy as np
from tqdm import trange

import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import picograd
import picograd.nn as nn
import picograd.optim as optim
from picograd.loss import CrossEntropyLoss
from picograd.draw_utils import draw_dot


block_size = 256
batch_size = 64
lr = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200

n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2


class GPT(nn.Module):
  def __init__(
    self,
    vocab_size: int,
    n_embed: int,
    block_size: int,
  ):
    super().__init__()

    self.token_embedding_table= nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    self.blocks = nn.Sequential(*[nn.TransformerBlock(n_embed, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embed) # final layer norm
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    token_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.position_embedding_table(picograd.Tensor(np.arange(T), device=device)) # (T, C)

    x = token_emb + pos_emb   # (B,T,C)
    x = self.blocks(x)        # (B, T, C)
    x = self.ln_f(x)          # (B, T, C)
    logits = self.lm_head(x)  # (B,T,vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C) 
      targets = targets.view(B*T)
      loss = CrossEntropyLoss(logits, targets)

    return logits, loss

def get_batch(data, batch_size, device):
  ix = torch.randint(len(data) - block_size, (batch_size,)).numpy()
  x = picograd.Tensor.stack([data[i:i+block_size] for i in ix])
  y = picograd.Tensor.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y


def tokenize_text(chars):
  stoi = { ch:i for i, ch in enumerate(chars) }
  itos = { i:ch for i, ch in enumerate(chars) }
  encode = lambda s: [stoi[c] for c in s]           # encoder: string => list of integers
  decode = lambda l: ''.join([itos[i] for i in l])  # decoder: list of integers => string
  return encode, decode

# @torch.no_grad()
def estimate_loss(model):
  out = {}
  model.eval()
  losses = picograd.Tensor.zeros(eval_iters)
  for k in range(eval_iters):
    X, Y = get_batch(train_data, batch_size)
    logits, loss = model(X, Y)
    losses[k] = loss.item()
  out['train'] = losses.mean()

  losses = torch.zeros(eval_iters)
  for k in range(eval_iters):
    X, Y = get_batch(val_data, batch_size)
    logits, loss = model(X, Y)
    losses[k] = loss.item()
  out['val'] = losses.mean()
  model.train()
  return out

def train_model(model):
  optimizer = optim.AdamW(model.get_params(), lr=lr)
  for iter in (t := trange(max_iters)):
    xb, yb = get_batch(train_data, batch_size, device)

    # if iter % eval_interval == 0:
    #   losses = estimate_loss(model)
    #   print(f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}")

    logits, loss = model(xb, yb)
    # draw_dot(loss, path="graphs/gpt_loss_graph")
    optimizer.zero_grad() # TODO: (set_to_none=True)
    loss.backward()
    optimizer.step()

    t.set_description(f"Loss: {loss.item()}")
  print(loss.item())


if __name__ == "__main__":
  device = picograd.Device(picograd.Devices.CPU)
  print("[*] Using device", device.name)

  with open("examples/gpt_input.txt", "r", encoding="utf-8") as f:
    text = f.read()
  chars = sorted(list(set(text)))
  vocab_size = len(chars)

  encode, decode = tokenize_text(chars)

  data = picograd.Tensor(encode(text), dtype=np.int64)
  block_size = 8            # context length
  n = int(0.9 * len(data))  # 90/10 train/test split
  train_data = data[:n]
  val_data = data[n:]

  # xb, yb = get_batch(train_data, batch_size, device)
  # for b in range(batch_size):
  #   for t in range(block_size):
  #     context = xb[b,:t+1]
  #     target = yb[b,t]
  #     print(f"when input is {context.tolist()} the target is {target.item()}")

  m = GPT(vocab_size, n_embed, block_size).to(device)
  train_model(m)

  # context = torch.zeros((1, 1), dtype=torch.long, device=device)
  # print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
