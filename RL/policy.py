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
                 pooler_config):
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

        self.pooler = getattr(poolers, pooler_config['name'])(pooler_config)

    def _convert_slots_to_rep(self, slots):
        return self.pooler(slots)

    def get_value(self, slots):
        return self.critic_net(self._convert_slots_to_rep(slots))
    
    def get_action_distribution(self, slots):
        actor_out = self.actor_net(self._convert_slots_to_rep(slots))
        if self.is_action_discrete:
            return Categorical(logits = actor_out)
        else:
            std = torch.exp(self.logstd.expand_as(actor_out))
            return Normal(loc = actor_out, scale = std)
    
    def get_action_logprob_entropy(self, slots, action = None):
        dist = self.get_action_distribution(slots)
        if action is None:
            action = dist.sample()
        # TODO check if log_prob is implemented correctly, 
        # meaning in discrete case it must return batch of logprobs and batch of entropies
        if self.is_action_discrete:
            return action, dist.log_prob(action), dist.entropy()
        else:
            return action, dist.log_prob(action).sum(1), dist.entropy().sum(1)
