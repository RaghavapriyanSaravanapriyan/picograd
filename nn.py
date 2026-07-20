import random
from engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
class Neuron(Module):
    def __init__(self, nin, linearity = True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.linearity = linearity

    def __call__(self, x): #dunder method for f(x)
        out = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return out.relu() if self.linearity else out
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'ReLU ' if self.linearity else 'Linear'}Neuron({len(self.w)})"

n1 = Neuron(5)
print(n1)

