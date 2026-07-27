from picograd.engine import Value
from picograd.nn import Neuron
from picograd.graph import graphy

n = Neuron(2)
x = [Value(3.0), Value(-2.0)]
y = n(x) #forward pass

y.backward() #backprop

g = graphy(y)
g.render("test", format="svg", cleanup=True)