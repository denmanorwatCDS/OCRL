from torch import nn

class IdentityPooler(nn.Module):
    def __init__(self, ocr_rep_dim, num_slots, config):
        super().__init__()

    def forward(self, inp):
        return inp