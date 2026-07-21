class Value:

    def __init__(self, data, _previous = (), _operation = ''):
        self.data = data
        self.grad = 0.0

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
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), 'pow')

        def _backward():
            self.grad += (other*self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data<0 else self.data, (self, ), 'relu')

        def _backward():
            self.grad += (out.data>0)*out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        #lets build the topological sort to traverse the nodes
        topo = []
        visited = set()
        def build_graph(node):
            if node not in visited:
                visited.add(node)
                for prev in node._prev:
                    build_graph(prev)
                topo.append(node)
        build_graph(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
