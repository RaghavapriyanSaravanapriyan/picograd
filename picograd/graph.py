from graphviz import Digraph

from picograd.engine import Value



def trace(root):

    visited = set()

    edges = set()



    def build(root):

        if root not in visited:

            visited.add(root)

            for i in root._prev:

                edges.add((i, root))

                build(i)

    build(root)

    return visited, edges



def graphy(root):

    dot = Digraph(format = 'svg', graph_attr={"rankdir":"LR"})



    nodes, edges = trace(root)



    for n in nodes:

        uid = str(id(n)) #id returns unique id of the object



        dot.node(uid, label=f"{{ {n.label} | data={n.data:.4f} | grad={n.grad:.4f} }}",shape="record",)



        if n._op:

            op = uid + n._op

            dot.node(op, label=n._op)

            dot.edge(op, uid)



    for n1, n2 in edges:

        dot.edge(str(id(n1)), str(id(n2)) + n2._op)



    return dot