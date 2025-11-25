import torch
from networks.utils.mlp import MultiHeadedMLPModule
from torch import nn

class ReturnPredictor(nn.Module):
    def __init__(self, feature_length, action_length, 
                 account_for_action, mlp_config):
        super().__init__()
        self.feature_length = feature_length
        self.account_for_action = account_for_action
        self.action_length = action_length
        self.net = MultiHeadedMLPModule(n_heads = 1, input_dim = self.in_dim, output_dims = 1,
                                        **mlp_config)

    def _fetch_data(self, feature_vec, action):
        if self.account_for_action:
            return torch.cat([feature_vec, action], dim = -1)
        return torch.cat([feature_vec], dim = -1)

    def forward(self, feature_vec, action = None):
        data = self._fetch_data(feature_vec, action)
        return self.net(data)[0]
    
    @property
    def in_dim(self):
        if self.account_for_action:
            return self.feature_length + self.action_length
        return self.feature_length