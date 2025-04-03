import torch
from torch import nn
from utils.layer_preprocessing.spectral_norm import spectral_norm
from networks.feature_extractors.base_networks import NormLayer

class DummySlotExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.outp_dim = input_dim
        assert input_dim.ndim == 0, 'Dummy works only for state-based environments'

    def forward(self, obs):
        return obs

class CNNExtractor(nn.Module):
    def __init__(self, channels, obs_size, act=nn.ELU, norm='none', cnn_depth=48, cnn_kernels=(4, 4, 4, 4), 
                 spectral_normalization=False):
        super().__init__()
        assert len(obs_size) == 2, 'CNN extractor works only for images'
        self._num_inputs = channels
        self._act = act()
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels

        self._conv_model = []
        for i, kernel in enumerate(self._cnn_kernels):
            if i == 0:
                prev_depth = channels
            else:
                prev_depth = 2 ** (i - 1) * self._cnn_depth
            depth = 2 ** i * self._cnn_depth
            if spectral_normalization:
                self._conv_model.append(spectral_norm(nn.Conv2d(prev_depth, depth, kernel, stride=2)))
            else:
                self._conv_model.append(nn.Conv2d(prev_depth, depth, kernel, stride=2))
            self._conv_model.append(NormLayer(norm, depth))
            self._conv_model.append(self._act)
        self._conv_model = nn.Sequential(*self._conv_model)
        
        self.outp_dim = torch.prod(torch.tensor(self.forward(torch.zeros((1, channels, *obs_size))).shape[1:]))

    def forward(self, data):
        output = self._conv_model(data)
        output = output.reshape(output.shape[0], 1, -1)
        return output