import torch
from networks.utils.mlp import MultiHeadedMLPModule
from torch import nn

class ReturnPredictor(nn.Module):
    def __init__(self, obs_length, task_length, 
                 action_length, account_for_action,
                 mlp_config):
        super().__init__()
        self.obs_length = obs_length
        self.task_length = task_length
        self.account_for_action = account_for_action
        self.action_length = action_length
        self.net = MultiHeadedMLPModule(n_heads = 1, input_dim = self.in_dim, output_dims = 1,
                                        **mlp_config)

    def _fetch_data(self, obs, task, action):
        if self.account_for_action:
            return torch.cat([obs, task, action], dim = -1)
        return torch.cat([obs, task], dim = -1)

    def forward(self, obs, task, action = None):
        data = self._fetch_data(obs, task, action)
        return self.net(data)[0]
    
    @property
    def in_dim(self):
        if self.account_for_action:
            return self.obs_length + self.task_length + self.action_length
        return self.obs_length + self.task_length