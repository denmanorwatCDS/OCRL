from torch import nn
from copy import deepcopy

class Identity(nn.Module):
    def __init__(self, ocr_config, env_config):
        super(Identity, self).__init__()

    def forward(self, inp):
        return inp
    
    def get_loss(self, obs, future_obs, do_dropout):
        return 0
    
    def get_grouped_parameters(self):
        return {'identity': self.named_parameters()}
    
    def training_mode(self):
        self.train()
        for param in self.parameters():
            param.requires_grad_ = True

    def inference_mode(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad_ = False

    def get_paramwise_lr(self):
        paramwise_lr = {}
        for name, _ in self.get_grouped_parameters().items():
            paramwise_lr[name] = None
        return paramwise_lr
    
    def get_slots(self, obs, training):
        return obs
    
    def requires_ppg(self):
        return False