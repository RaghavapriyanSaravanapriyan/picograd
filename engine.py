class Value:

    def __init__(self, data, _previous = (), _operation = ''):
        self.data = data
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(_previous) # set of previous nodes
        self._op = _operation # operation with which this node came about
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad+=out.grad # gradients flow in addition
            other.grad+=out.grad
        out._backward = _backward

        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other), '*')

        def _backward():
            self.grad+=other.data*out.grad #local derivative * incoming derivative
            other.grad+=self.data*out.grad
        out._backward = _backward

        return out
    def relu(self):
        out = Value(0 if self.data<0 else self.data, (self, ), 'relu')
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
a = Value(1)
b = Value(2)
d = Value(3)
print(a, b)
c = a*b + d
print(c.data)
c.grad = 1
c._backward()
print(a.grad, b.grad, d.grad)
