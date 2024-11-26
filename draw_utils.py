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

# FIXME
def draw_dot(root, format='svg', rankdir='LR'):
  assert rankdir in ['LR', 'TB']
  nodes, edges = trace(root)
  dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
  
  for n in nodes:
    #dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
    dot.node(name=str(id(n)), label = f"[ data {str(n.data)} | grad {n.grad} ]", shape='record')
    #if n._op:
      #dot.node(name=str(id(n)) + n._op, label=n._op)
      #dot.edge(str(id(n.name)) + n._op, str(id(n.name)))
  
  for n1, n2 in edges:
    #dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    dot.edge(str(id(n1)), str(id(n2)))

  dot.render('graphs/output')
  return dot