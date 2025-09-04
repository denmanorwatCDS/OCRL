from torch import nn

class IdentityPooler(nn.Module):
    def __init__(self, pooler_config):
        super().__init__()

    def forward(self, inp):
        return inp