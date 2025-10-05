import math, torch, functools
from utils.distributions.tanh import TanhNormal, PPOTanhNormal
from torch.nn.init import orthogonal_, xavier_normal_, zeros_

def fetch_activation(activation_name):
    assert activation_name in ['relu', 'tanh', 'elu'], 'Only relu, elu or tanh are supported as activations'
    if activation_name == 'relu':
        return torch.nn.ReLU
    elif activation_name == 'tanh':
        return torch.nn.Tanh
    elif activation_name == 'elu':
        return torch.nn.ELU
    
def fetch_dist_type(class_name):
    assert class_name in ['TanhNormal', 'PPOTanhNormal']
    if class_name == 'TanhNormal':
        return TanhNormal
    if class_name == 'PPOTanhNormal':
        return PPOTanhNormal
    
def fetch_init(name, gain = None):
    if name == 'xavier':
        if gain is None:
            gain = 1.
        return functools.partial(xavier_normal_, gain = gain)
    elif name == 'orthogonal':
        if gain is None:
            gain = math.sqrt(2)
        return functools.partial(orthogonal_, gain = gain)
    elif name == 'zeros':
        return zeros_