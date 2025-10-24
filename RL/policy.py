import torch
import numpy as np
from torch import nn
from RL import poolers
from torch.distributions import Categorical, Normal
from functools import partial

def ortho_linear_init(layer, std = np.sqrt(2), bias_const = 0.):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain = std)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, bias_const)

def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def fetch_activation(act_name):
    return getattr(nn, act_name)()

class Policy(nn.Module):
    def __init__(self, 
                 observation_size, action_size, is_action_discrete,
                 backbone_mlp, backbone_act, actor_mlp, actor_act, critic_mlp, critic_act,
                 pooler_config, ocr_rep_dim, num_slots):
        super().__init__()

        self.is_action_discrete = is_action_discrete
        assert isinstance(observation_size, int) and isinstance(action_size, int)
        self.pooler = getattr(poolers, pooler_config['name'])(config = pooler_config, 
                                                              ocr_rep_dim = ocr_rep_dim, num_slots = num_slots)
        
        self.pooler.apply(partial(ortho_linear_init, std = np.sqrt(2)))
        self.backbone_net = self.fetch_module_by_kwargs(in_dim = self.pooler.rep_dim, mlp_architecture = backbone_mlp, 
                                                        activation = backbone_act)
        self.actor_net = self.fetch_module_by_kwargs(in_dim = backbone_mlp[-1], mlp_architecture = actor_mlp, 
                                                     activation = actor_act, output_dim = action_size, output_std = 0.01)
        self.critic_net = self.fetch_module_by_kwargs(in_dim = backbone_mlp[-1], mlp_architecture = critic_mlp, 
                                                      activation = critic_act, output_dim = 1, output_std = 1.)

        if not is_action_discrete:
            self.logstd = nn.Parameter(torch.zeros(1, np.prod(action_size)))

    def fetch_module_by_kwargs(self, in_dim, mlp_architecture, activation, output_dim = None, output_std = None):
        modules = []
        modules.append(layer_init(nn.Linear(in_dim, mlp_architecture[0])))
        for in_dim, out_dim in zip(mlp_architecture[:-1], mlp_architecture[1:]):
            modules.append(fetch_activation(activation))
            modules.append(layer_init(nn.Linear(in_dim, out_dim)))
        modules.append(fetch_activation(activation))
        if output_dim is not None:
            modules.append(layer_init(nn.Linear(mlp_architecture[-1], output_dim), std = output_std))
        return nn.Sequential(*modules)

    def _convert_slots_to_rep(self, slots):
        return self.pooler(slots)

    def get_value(self, slots):
        feat = self.backbone_net(self._convert_slots_to_rep(slots))
        return self.critic_net(feat).squeeze(axis = -2)
    
    def get_action_distribution(self, slots):
        feat = self.backbone_net(self._convert_slots_to_rep(slots))
        actor_out = self.actor_net(feat)
        actor_out = torch.unsqueeze(actor_out, dim = 1)
        if self.is_action_discrete:
            return Categorical(logits = actor_out)
        else:
            std = torch.exp(self.logstd.expand_as(actor_out))
            return Normal(loc = actor_out, scale = std)
    
    # TODO check logit shape. It must be of the shape: batch x actions.
    def get_action_logprob_entropy(self, slots, action = None):
        dist = self.get_action_distribution(slots)
        detached_dist = self.get_action_distribution(slots.detach())
        if action is None:
            action = dist.sample()
        elif not (action is None) and not self.is_action_discrete:
            # TODO check if unsqueeze is required
            action = torch.unsqueeze(action, axis = -2)
        # TODO check if log_prob is implemented correctly, 
        # meaning in discrete case it must return batch of logprobs and batch of entropies
        if self.is_action_discrete:
            return action.squeeze(dim = -2), dist.log_prob(action).squeeze(dim = -2),\
                detached_dist.entropy().squeeze(dim = -2)
        else:
            return action.squeeze(dim = -2), dist.log_prob(action).sum(-1).squeeze(dim = -2),\
                detached_dist.entropy().sum(-1).squeeze(dim = -2)
        
    def get_paramwise_lr(self):
        return {'policy': None}
    
    def inference_mode(self):
        self.eval()
        for name, param in self.named_parameters():
            param.requires_grad = False

    def training_mode(self):
        self.train()
        for name, param in self.named_parameters():
            param.requires_grad = True