from torch import nn
from copy import deepcopy

class Identity(nn.Module):
    def __init__(self, ocr_config, env_config):
        super().__init__()

    def forward(self, inp):
        return inp
    
    def get_grouped_parameters(self):
        return {'identity': self.named_parameters()}
    
    def training_mode(self):
        super().training_mode()
        self._prepare_enc()

    def get_paramwise_lr(self):
        paramwise_lr = {}
        for name, _ in self.get_grouped_parameters().items():
            paramwise_lr[name] = None
        return paramwise_lr