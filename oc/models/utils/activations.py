import torch
from torch.nn import Tanh

class Ball(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x):
        clipping_needed = torch.linalg.vector_norm(x, ord = 2, dim = -1, keepdim = True) > 1.
        return torch.where(clipping_needed, x / torch.linalg.vector_norm(x, ord = 2, dim = -1, 
                                                                         keepdim = True), x)
    
class Sphere(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x):
        return x / torch.linalg.vector_norm(x, ord = 2, dim = -1, keepdim = True)