from torch import nn

def mlp_builder(in_dim, net_architecture, out_dim, nonlinearity_name):
    modules = []
    layers = [*net_architecture, out_dim]
    NonLinearity = getattr(nn, nonlinearity_name)
    modules.append(nn.Linear(in_dim, net_architecture[0]))
    for in_dim, out_dim in zip(layers[:-1], layers[1:]):
        modules.append(NonLinearity())
        modules.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*modules)