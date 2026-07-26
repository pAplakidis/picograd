from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from picograd.backend.function import OPS
if TYPE_CHECKING: from picograd.tensor import Tensor


@dataclass(eq=False)
class ASTNode:
  op: OPS | None  # None for leaf node
  inputs: tuple[ASTNode, ...]
  tensor: Tensor  # owning Tensor
  forward_args: tuple = ()
  forward_kwargs: dict = None


def build_ast(tensor: Tensor, memo=None):
  if memo is None:
    memo = {}

  if tensor in memo:
    return memo[tensor]
  
  # Leaf tensor
  if tensor.prev_op is None or tensor.realized or len(tensor._prev) == 0:
    node = ASTNode(op=None, inputs=tuple(), tensor=tensor)
  else:
    # recursively build children
    inputs = tuple(build_ast(t, memo) for t in tensor._prev)
    node = ASTNode(op=tensor.prev_op, inputs=inputs, tensor=tensor, forward_args=tensor._prev_forward_args, forward_kwargs=tensor._prev_forward_kwargs)

  memo[tensor] = node
  return node
  
def linearize(root: ASTNode):
  visited = set()
  order = []

  def dfs(node):
    if node in visited:
      return
    visited.add(node)
    for inp in node.inputs:
      dfs(inp)
    order.append(node)

  dfs(root)
  return order

# TODO: might be useful for kernel fusion and scheduling
def compute_levels(node: ASTNode, levels=None):
  if levels is None:
    levels = {}

  if node in levels:
    return levels[node]

  if node.op is None:
    level = 0
  else:
    level = 1 + max(compute_levels(inp, levels) for inp in node.inputs)

  levels[node] = level
  return level
