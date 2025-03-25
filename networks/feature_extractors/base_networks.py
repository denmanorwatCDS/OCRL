import torch
from torch import nn
from utils.layer_preprocessing.spectral_norm import spectral_norm

class NormLayer(nn.Module):
    def __init__(self, name, dim=None):
        super().__init__()
        if name == 'none':
            self._layer = None
        elif name == 'layer':
            assert dim != None
            self._layer = nn.LayerNorm(dim)
        else:
            raise NotImplementedError(name)

    def forward(self, features):
        if self._layer is None:
            return features
        return self._layer(features)