from graphviz import Digraph

def trace(root):
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in getattr(v, "_prev", []):  # if v is scalar, _prev may not exist
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

from graphviz import Digraph

def trace(root):
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in getattr(v, "_prev", []):
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root, format='png', rankdir='TB', verbose=False, path="graphs/output"):
  nodes, edges = trace(root)
  dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

  for n in nodes:
    # --- handle plain Python floats/ints (constants) ---
    if isinstance(n, (float, int)):
      dot.node(
        name=str(id(n)),
        label=str(n),
        shape="circle",
        style="filled",
        fillcolor="lightblue"
      )
      continue

    name = getattr(n, "name", "")

    # --- build data and grad strings ---
    data_str = str(getattr(n, "_data", "")) if verbose else "data"
    grad_str = str(getattr(n, "_grad", "")) if verbose and getattr(n, "requires_grad", False) else "grad"

    # --- build node label ---
    if getattr(n, "requires_grad", False):
      # grad row only added if requires_grad=True
      label = f"{{ {name} | {data_str} | {grad_str} }}"
    else:
      label = f"{{ {name} | {data_str} }}"  # no grad row

    # --- create node ---
    dot.node(
      name=str(id(n)),
      label=label,
      shape='record',
      style='filled' if not getattr(n, "requires_grad", False) else '',
      fillcolor='lightgrey' if not getattr(n, "requires_grad", False) else ''
    )

    # --- ops ---
    if getattr(n, "prev_op", None):
      dot.node(
        name=str(id(n)) + str(n.prev_op),
        label=str(n.prev_op),
        shape="circle",
        style="filled",
        fillcolor="orange"
      )
      dot.edge(str(id(n)) + str(n.prev_op), str(id(n)))

  # --- edges ---
  for n1, n2 in edges:
    dot.edge(str(id(n1)), str(id(n2)) + str(getattr(n2, "prev_op", "")))

  dot.render(path)
  print("[+] Graph diagram saved at", path)
  return dot
