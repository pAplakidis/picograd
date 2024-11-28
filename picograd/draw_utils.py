from graphviz import Digraph

def trace(root):
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root, format='png', rankdir='TB'):
  """
  format: png | svg | ...
  rankdir: TB (top to bottom graph) | LR (left to right)
  """
  assert rankdir in ['LR', 'TB']
  nodes, edges = trace(root)
  dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
  
  for n in nodes:
    dot.node(name=str(id(n)), label = f"{{ data {n.data} | grad {n.grad} }}", shape='record')
    if n.prev_op:
      dot.node(name=str(id(n)) + str(n.prev_op), label=str(n.prev_op))
      dot.edge(str(id(n)) + str(n.prev_op), str(id(n)))
  
  for n1, n2 in edges:
    # FIXME: doesn't add 
    dot.edge(str(id(n1)), str(id(n2)) + str(n2.prev_op))
    # dot.edge(str(id(n1)), str(id(n2)))

  dot.render('graphs/output')
  return dot