import torch
import numpy as np
from torch import nn
from RL import poolers
from torch.distributions import Categorical, Normal

def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def fetch_activation(act_name):
    return getattr(nn, act_name)()

class Policy(nn.Module):
    def __init__(self, 
                 observation_size, action_size, is_action_discrete,
                 actor_mlp, actor_act, critic_mlp, critic_act,
                 pooler_config, ocr_rep_dim, num_slots):
        super().__init__()

        self.is_action_discrete = is_action_discrete
        assert isinstance(observation_size, int) and isinstance(action_size, int)
        actor_modules, critic_modules = [], []
        actor_modules.append(layer_init(nn.Linear(observation_size, actor_mlp[0])))
        for in_dim, out_dim in zip(actor_mlp[:-1], actor_mlp[1:]):
            actor_modules.append(fetch_activation(actor_act))
            actor_modules.append(layer_init(nn.Linear(in_dim, out_dim)))
        actor_modules.append(fetch_activation(actor_act))
        actor_modules.append(layer_init(nn.Linear(out_dim, action_size), std = 0.01))
        
        critic_modules.append(layer_init(nn.Linear(observation_size, critic_mlp[0])))
        for in_dim, out_dim in zip(critic_mlp[:-1], critic_mlp[1:]):
            critic_modules.append(fetch_activation(critic_act))
            critic_modules.append(layer_init(nn.Linear(in_dim, out_dim)))
        critic_modules.append(layer_init(nn.Linear(out_dim, 1), std = 1.))

        self.actor_net = nn.Sequential(*actor_modules)
        self.critic_net = nn.Sequential(*critic_modules)

        if not is_action_discrete:
            self.logstd = nn.Parameter(torch.zeros(1, np.prod(action_size)))

        self.pooler = getattr(poolers, pooler_config['name'])(config = pooler_config, 
                                                              ocr_rep_dim = ocr_rep_dim, num_slots = num_slots)

    def _convert_slots_to_rep(self, slots):
        return self.pooler(slots)

    def get_value(self, slots):
        return self.critic_net(self._convert_slots_to_rep(slots)).squeeze(axis = -2)
    
    def get_action_distribution(self, slots):
        actor_out = self.actor_net(self._convert_slots_to_rep(slots))
        actor_out = torch.unsqueeze(actor_out, dim = 1)
        if self.is_action_discrete:
            return Categorical(logits = actor_out)
        else:
            std = torch.exp(self.logstd.expand_as(actor_out))
            return Normal(loc = actor_out, scale = std)
    
    # TODO check logit shape. It must be of the shape: batch x actions
    def get_action_logprob_entropy(self, slots, action = None):
        dist = self.get_action_distribution(slots)
        if action is None:
            action = dist.sample()
        elif not (action is None) and not self.is_action_discrete:
            action = torch.unsqueeze(action, axis = -2)
        # TODO check if log_prob is implemented correctly, 
        # meaning in discrete case it must return batch of logprobs and batch of entropies
        if self.is_action_discrete:
            return action.squeeze(dim = -2), dist.log_prob(action).squeeze(dim = -2),\
                dist.entropy().squeeze(dim = -2)
        else:
            return action.squeeze(dim = -2), dist.log_prob(action).sum(-1).squeeze(dim = -2),\
                dist.entropy().sum(-1).squeeze(dim = -2)
        
    def get_paramwise_lr(self):
        return {'policy': None}
    
    def inference_mode(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad_ = False

    def training_mode(self):
        self.train()
        for param in self.parameters():
            param.requires_grad_ = True