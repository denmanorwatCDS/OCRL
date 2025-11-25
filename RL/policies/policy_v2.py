import torch
import numpy as np
from networks.distributions.distributions import get_distribution_network
from networks.regressors.regressors import ReturnPredictor
from utils.distributions.tanh import TanhNormal

class Policy(torch.nn.Module):
    def __init__(self,
                 name, device,
                 feature_len, action_length, account_for_action,
                 *,
                 actor_config,
                 critic_config,
                 clip_action=False,
                 force_use_mode_actions=False,
                 ):
        self.name = name
        self.actor = get_distribution_network(name = actor_config.distribution_name, feature_length = feature_len, 
                                              output_dim = action_length,
                                              distribution_config = actor_config.distribution).to(device)
        self.critic1 = ReturnPredictor(feature_length=feature_len, action_length = action_length, 
                                      account_for_action = account_for_action, mlp_config = critic_config.mlp).to(device)
        self.critic2 = ReturnPredictor(feature_length=feature_len, action_length = action_length, 
                                      account_for_action = account_for_action, mlp_config = critic_config.mlp).to(device)
        self.device = device
        self._clip_action = clip_action
        self._force_use_mode_actions = force_use_mode_actions

    def _move_to_torch(self, observations, tasks):
        if not isinstance(observations, torch.Tensor):
            observations = torch.as_tensor(observations).to(next(self.parameters()).device)
        if not isinstance(tasks, torch.Tensor):
            tasks = torch.as_tensor(tasks).to(next(self.parameters()).device)
        return observations, tasks
    
    def forward(self, single_features):
        dist = self.actor(single_features)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()
        if hasattr(dist, '_normal'):
            info.update(dict(
                normal_mean=dist._normal.mean,
                normal_std=dist._normal.variance.sqrt(),
            ))

        return dist, info
    
    def forward_mode(self, single_features):
        samples = self.actor.forward_mode(single_features)
        return samples, dict()

    def get_mode_actions(self, single_features):
        with torch.no_grad():
            samples, info = self.forward_mode(single_features)
            return samples.cpu().numpy(), {
                k: v.detach().cpu().numpy()
                for (k, v) in info.items()
            }

    def get_sample_actions(self, single_features):
        with torch.no_grad():
            dist, info = self.forward(single_features)
            if isinstance(dist, TanhNormal):
                pre_tanh_values, actions = dist.rsample_with_pre_tanh_value()
                log_probs = dist.log_prob(actions, pre_tanh_values)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: v.detach().cpu().numpy()
                    for (k, v) in info.items()
                }
                infos['pre_tanh_value'] = pre_tanh_values.detach().cpu().numpy()
                infos['log_prob'] = log_probs.detach().cpu().numpy()
            else:
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: v.detach().cpu().numpy()
                    for (k, v) in info.items()
                }
                infos['log_prob'] = np.squeeze(log_probs.detach().cpu().numpy())
            return actions, infos

    def get_actions(self, observations, tasks, obj_idxs):
        observations, obj_idxs = torch.from_numpy(observations).to(self.device), torch.from_numpy(obj_idxs).to(self.device)
        tasks = torch.from_numpy(tasks).to(self.device)
        features = self.pooler(observations, tasks, obj_idxs)
        if self._force_use_mode_actions:
            actions, info = self.get_mode_actions(features)
        else:
            actions, info = self.get_sample_actions(features)
        if self._clip_action:
            epsilon = 1e-6
            actions = np.clip(
                actions,
                self.env_spec.action_space.low + epsilon,
                self.env_spec.action_space.high - epsilon,
            )
        return actions, info
        
    def get_logprob_and_entropy(self, single_features, actions, pre_tanh_actions):
        dist, info = self.forward(single_features = single_features)
        log_probs, entropy = dist.log_prob(value = actions, pre_tanh_value = pre_tanh_actions), dist.entropy()
        return np.squeeze(log_probs), entropy, info