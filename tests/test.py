import torch
from engine import Value

def test_engine():
    a = Value(5)

    b = a * 10 + a**2
    c = b.relu() + b*2
    d = c + b * a.relu()
    e = d**2
    e.backward()
    
    apg, epg = a, e

    a = torch.tensor([5.0]).double()
    a.requires_grad = True
    b = a * 10 + a**2
    c = b.relu() + b*2
    d = c + b * a.relu()
    e = d**2
    e.backward()

    apt, ept = a, e

    # forward pass went well
    assert epg.data == ept.data.item()

    # backward pass went well
    assert apg.grad == apt.grad.item()
    