class Value:

    def __init__(self, data, _previous = (), _operation = ''):
        self.data = data
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(_previous) # set of previous nodes
        self._op = _operation # operation with which this node came about
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
a = Value(1)
b = Value(2)
print(a, b)
c = a+b
print(c._prev)
