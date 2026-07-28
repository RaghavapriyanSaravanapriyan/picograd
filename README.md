# picograd 


![image](https://github.com/RaghavapriyanSaravanapriyan/picograd/blob/main/graph.svg)


picograd is a tiny scalar based autograd engine written in pure Python (~110 lines) heavily inspired from Andrej Karpathy's beautiful micrograd implementation. The engine calculates the gradients of each scalar by backpropagation which is implemented from the ground up. The engine makes a Directed Acyclic Graph [DAG] while using a topological sort algorithm to traverse all the nodes and calculate gradients for each Value object by using the simple principles of differential calculus. 

Picograd also includes a neural-network library implemented by stacking Value objects into neurons, neurons into layers, then layers into a full blown MLP. The MLP was then trained on make_circles dataset limited to about 100 training examples to test out the implementation. 

## Example usage of picograd

```python
from picograd.engine import Value

x1 = Value(2.5)
x2 = Value(-3.0)
w1 = Value(-1.5)
w2 = Value(0.7)
b  = Value(1.0)

n = x1 * w1 + x2 * w2 + b
o = n.sigmoid()

p = o**2 + (x1 - x2).relu()
q = p.log()
r = 6.0 / (w1 * w2) + q * 3

print(f'{r.data:.4f}')  # -0.6000
r.backward()
print(f'{w1.grad:.4f}')  # -3.8094
print(f'{w2.grad:.4f}')  # 8.1631
print(f'{b.grad:.4f}')   # 0.0001
```

## Single neuron backprop 

![image](https://github.com/RaghavapriyanSaravanapriyan/picograd/blob/main/tests/test.svg)

```python
from picograd.engine import Value
from picograd.nn import Neuron
from picograd.graph import graphy

# Create a neuron with 2 inputs
n = Neuron(2)

# Input values
x = [Value(3.0), Value(-2.0)]

# Forward pass
y = n(x)

# Backward pass
y.backward()

# Visualize the computation graph
g = graphy(y)
g.render("test", format="svg", cleanup=True)
```

## Run test.py

To run test.py you will have to install pytorch and pytest

```python
pytest tests/test.py
```
