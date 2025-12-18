import torch
import numpy as np
from networks.distributions.distributions import get_distribution_network
from networks.utils.mlp_builder import mlp_builder
from utils.distributions.tanh import TanhNormal
from utils.config_to_module import fetch_dist_type
from torch.distributions.independent import Independent

class Actor(torch.nn.Module):
    def __init__(self,
                 feature_len, action_length,
                 *,
                 distribution_class,
                 distribution_parameterization,
                 hidden_sizes, hidden_nonlinearity,
                 init_std,
                 clip_action = False,
                 force_use_mode_actions = False,
                 ):
        super().__init__()
        self.calculate_mean_std = self._get_mean_std_calculator(distribution_parameterization = distribution_parameterization,
                                                                in_dim = feature_len, output_dim = action_length,
                                                                hidden_sizes = hidden_sizes,
                                                                hidden_nonlinearity = hidden_nonlinearity,
                                                                init_std = init_std)
        self.distribution_class = fetch_dist_type(distribution_class)

        self._clip_action = clip_action
        self._force_use_mode_actions = force_use_mode_actions

    def _get_mean_std_calculator(self, distribution_parameterization, in_dim, output_dim, 
                                 hidden_sizes, hidden_nonlinearity, init_std):
        self.register_buffer(name='init_std', tensor = torch.log(torch.full((output_dim,), init_std)))
        if distribution_parameterization == 'GlobalStd':
            self.mean_net = mlp_builder(in_dim = in_dim, net_architecture = hidden_sizes, 
                                        out_dim = output_dim, nonlinearity_name = hidden_nonlinearity)
            self.logstd = torch.nn.Parameter(torch.zeros((output_dim,))) + self.init_std
            self.mean_net, self.logstd = self.mean_net, self.logstd
            
            def calculate_mean_std(inp):
                return self.mean_net(inp), torch.exp(self.logstd)
        
        elif distribution_parameterization == 'TwoHeads':
            self.mean_logstd_net = mlp_builder(in_dim = in_dim, net_architecture = hidden_sizes,
                                               out_dim = 2 * output_dim, nonlinearity_name = hidden_nonlinearity)

            def calculate_mean_std(inp):
                mean, logstd = torch.split(self.mean_logstd_net(inp), split_size_or_sections = output_dim, dim = -1)
                logstd = logstd + self.init_std
                return mean, torch.exp(logstd)
        
        elif distribution_parameterization == 'TwoNetworks':
            self.mean_net = mlp_builder(in_dim = in_dim, net_architecture = hidden_sizes, 
                                   out_dim = output_dim, nonlinearity_name = hidden_nonlinearity)
            self.logstd_net = mlp_builder(in_dim = in_dim, net_architecture = hidden_sizes,
                                          out_dim = output_dim, nonlinearity_name = hidden_nonlinearity)
            self.mean_net, self.logstd_net = self.mean_net, self.logstd_net
            
            def calculate_mean_std(inp):
                logstd = self.logstd_net(inp)
                logstd = logstd + self.init_std
                return self.mean_net(inp), torch.exp(logstd)

        return calculate_mean_std
    
    def get_dist(self, inp):
        mean, std = self.calculate_mean_std(inp)
        dist = self.distribution_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)
        return dist
    
    def get_mode(self, inp):
        mean, _ = self.calculate_mean_std(inp)
        return mean

    def forward(self, single_features):
        dist = self.get_dist(single_features)
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
        samples = self.get_mode(single_features)
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

    def get_actions(self, single_features):
        if self._force_use_mode_actions:
            actions, info = self.get_mode_actions(single_features)
        else:
            actions, info = self.get_sample_actions(single_features)
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