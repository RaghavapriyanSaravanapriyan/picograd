from picograd.graph import graphy
from picograd.engine import Value

a = Value(2, label= 'a')
b = Value(3, label='b')
c = a*b
c.label = 'c'

g = graphy(c)
g.render("graph", format="svg", view=True)