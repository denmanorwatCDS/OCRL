import torch
from networks.utils.mlp import MultiHeadedMLPModule
from torch import nn
from networks.utils.mlp_builder import mlp_builder


class ReturnPredictor(nn.Module):
    def __init__(self, feature_length, action_length, 
                 account_for_action, hidden_sizes, nonlinearity_name):
        super().__init__()
        self.feature_length = feature_length
        self.account_for_action = account_for_action
        self.action_length = action_length
        self.net = mlp_builder(in_dim = self.in_dim, net_architecture = hidden_sizes, out_dim = 1,
                               nonlinearity_name = nonlinearity_name)

    def _fetch_data(self, feature_vec, action):
        if self.account_for_action:
            return torch.cat([feature_vec, action], dim = -1)
        return torch.cat([feature_vec], dim = -1)

    def forward(self, feature_vec, action = None):
        data = self._fetch_data(feature_vec, action)
        return self.net(data)
    
    @property
    def in_dim(self):
        if self.account_for_action:
            return self.feature_length + self.action_length
        return self.feature_length