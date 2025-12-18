import copy
import torch
import torch.nn.functional as F
import numpy as np
from RL.policies.policy_v2 import Actor
from networks.poolers.poolers import get_pooler_network
from torch.optim import AdamW
from itertools import chain
from networks.utils.parameter import ParameterModule
from networks.regressors.regressors import ReturnPredictor

class SAC(torch.nn.Module):
    def __init__(self,
                 name,
                 obs_length, task_length, obj_qty, action_length,
                 actor_config, critic_config, pooler_config,
                 lr, wd,
                 clip_action=False,
                 force_use_mode_actions=False,
                 *,
                 alpha,
                 tau,
                 scale_reward,
                 env_spec,
                 target_coef,
                 device,
                 discount):
        super().__init__()
        self.pooler = get_pooler_network(name = pooler_config.name, obs_length = obs_length, skill_length = task_length, 
                                         obj_qty = obj_qty, pooler_config = pooler_config.kwargs).to(device)
        self.target_pooler = copy.deepcopy(self.pooler)
        
        self.log_alpha = torch.nn.Parameter(data = torch.log(torch.Tensor([alpha])).to(device),
                                            requires_grad = True)
        self.discount = discount
        self.device = device

        self.critic1 = ReturnPredictor(feature_length = self.pooler.outp_dim, action_length = action_length, 
                                       account_for_action = True, 
                                       nonlinearity_name = critic_config.hidden_nonlinearity,
                                       hidden_sizes = critic_config.hidden_sizes).to(device)
        self.critic2 = ReturnPredictor(feature_length = self.pooler.outp_dim, action_length = action_length, 
                                       account_for_action = True, 
                                       nonlinearity_name = critic_config.hidden_nonlinearity,
                                       hidden_sizes = critic_config.hidden_sizes).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.actor = Actor(feature_len = self.pooler.outp_dim, action_length = action_length,
                           distribution_class = actor_config.normal_distribution_cls, 
                           distribution_parameterization = actor_config.distribution_parameterization,
                           hidden_sizes = actor_config.hidden_sizes, 
                           hidden_nonlinearity = actor_config.hidden_nonlinearity,
                           init_std = actor_config.init_std, clip_action = clip_action,
                           force_use_mode_actions = force_use_mode_actions).to(device)
        
        self.tau = tau
        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(env_spec.action_space.shape).item() / 2. * target_coef
        self.optimizer = AdamW(params = self.parameters(), lr = lr, weight_decay = wd)

    @property
    def on_policy(self):
        return False
    
    def eval(self):
        self.actor.eval(), self.critic1.eval(), self.critic2.eval(),\
            self.target_critic1.eval(), self.target_critic2.eval(),\
                self.pooler.eval(), self.target_pooler.eval()

    def train(self):
        self.actor.train(), self.critic1.train(), self.critic2.train(),\
            self.target_critic1.train(), self.target_critic2.train(),\
                self.pooler.train(), self.target_pooler.train()
        
    def get_actions(self, observations, tasks, obj_idxs):
        observations, obj_idxs = torch.from_numpy(observations).to(self.device), torch.from_numpy(obj_idxs).to(self.device)
        tasks = torch.from_numpy(tasks).to(self.device)
        single_features = self.pooler(observations, tasks, obj_idxs)
        return self.actor.get_actions(single_features)

    def optimize_op(self, observations, next_observations, obj_idxs, options, actions, dones, rewards):
        logs = {}
        cur_features, next_features = self.pooler(observations, options, obj_idxs), self.pooler(next_observations, options, obj_idxs)
        target_next_features = self.target_pooler(next_observations, options, obj_idxs)
        loss_qf, qf_logs = self._update_loss_qf(
            cur_features = cur_features,
            actions = actions,
            next_features = next_features, target_next_features = target_next_features,
            dones = dones,
            rewards = rewards * self._reward_scale_factor
        )
        new_action_log_probs, sacp_loss, sacp_logs = self._update_loss_sacp(cur_features = cur_features)

        loss_alpha, alpha_logs = self._update_loss_alpha(new_action_log_probs)
        
        self.optimizer.zero_grad()
        loss = (loss_qf + sacp_loss + loss_alpha)
        loss.backward()
        self.optimizer.step()
        self._update_targets()
        
        logs.update({**qf_logs, **sacp_logs, **alpha_logs})
        return logs
    
    def inference_value(self, observations, actions, options, obj_idxs):
        with torch.no_grad():
            batch_length, horizon_length, obj_length = observations.shape[:3]
            observation_shape = observations.shape[3:]
            observations = torch.from_numpy(observations.reshape((batch_length * horizon_length, obj_length, *observation_shape))).to(self.device)
            actions = torch.from_numpy(actions.reshape((batch_length * horizon_length, -1))).to(self.device)
            options = torch.from_numpy(options.reshape((batch_length * horizon_length, -1))).to(self.device)
            obj_idxs = torch.from_numpy(obj_idxs.reshape((batch_length * horizon_length))).to(self.device)
            cur_features = self.pooler(observations, options, obj_idxs)
            values = torch.min(
                self.target_critic1(cur_features, actions).flatten(),
                self.target_critic2(cur_features, actions).flatten(),
            )
        return values.reshape((batch_length, horizon_length)).detach().cpu().numpy()
    
    def _optim_zero_grad(self):
        for key in self._optimizers.keys():
            self._optimizers[key].zero_grad()

    def _optim_step(self):
        for key in self._optimizers.keys():
            self._optimizers[key].step()

    def _update_targets(self):
        target_nets = [self.target_critic1, self.target_critic2, self.target_pooler]
        nets = [self.critic1, self.critic2, self.pooler]
        for target_net, net in zip(target_nets, nets):
            for t_param, param in zip(target_net.parameters(), net.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                                   param.data * self.tau)
                
    def disable_grad_calc(self, networks):
        for net in networks:
            for param in net.parameters():
                param.requires_grad_(False)

    def enable_grad_calc(self, networks):
        for net in networks:
            for param in net.parameters():
                param.requires_grad_(True)
                
    def _update_loss_qf(self,
        cur_features,
        actions,
        next_features, target_next_features,
        dones,
        rewards,
    ):
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        q1_pred = self.critic1(cur_features, actions).flatten()
        q2_pred = self.critic2(cur_features, actions).flatten()
        
        next_action_dists, *_ = self.actor(next_features)
        if hasattr(next_action_dists, 'rsample_with_pre_tanh_value'):
            new_next_actions_pre_tanh, new_next_actions = next_action_dists.rsample_with_pre_tanh_value()
            new_next_action_log_probs = next_action_dists.log_prob(new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)
        else:
            new_next_actions = next_action_dists.rsample()
            new_next_actions = self._clip_actions(new_next_actions)
            new_next_action_log_probs = next_action_dists.log_prob(new_next_actions)

        target_q_values = torch.min(
            self.target_critic1(target_next_features, new_next_actions).flatten(),
            self.target_critic2(target_next_features, new_next_actions).flatten(),
        )
        target_q_values = target_q_values - alpha * new_next_action_log_probs
        target_q_values = target_q_values * self.discount

        with torch.no_grad():
            q_target = rewards + target_q_values * (1. - dones.float())

        # critic loss weight: 0.5
        loss_qf1 = F.mse_loss(q1_pred, q_target) * 0.5
        loss_qf2 = F.mse_loss(q2_pred, q_target) * 0.5

        return loss_qf1 + loss_qf2, {
            'LossQf1': loss_qf1.detach(),
            'LossQf2': loss_qf2.detach(),
            'QTargetsMean': q_target.mean().detach(),
            'QTdErrsMean': (((q_target - q1_pred).mean() + (q_target - q2_pred).mean()) / 2).detach(),
        }
        
    def _update_loss_sacp(
            self, cur_features, 
    ):
        with torch.no_grad():
            alpha = self.log_alpha.exp()

        action_dists, *_ = self.actor(cur_features)
        if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
            new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
            new_action_log_probs = action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)
        else:
            new_actions = action_dists.rsample()
            new_actions = self._clip_actions(new_actions)
            new_action_log_probs = action_dists.log_prob(new_actions)
        
        self.disable_grad_calc([self.critic1, self.critic2])
        min_q_values = torch.min(
            self.critic1(cur_features.detach(), new_actions).flatten(),
            self.critic2(cur_features.detach(), new_actions).flatten(),
        )

        loss_sacp = (alpha * new_action_log_probs - min_q_values).mean()
        self.enable_grad_calc([self.critic1, self.critic2])
        
        return new_action_log_probs, loss_sacp, {
            'LossSacp': loss_sacp.detach(),
            'SacpNewActionLogProbMean': new_action_log_probs.mean()
        }

    def _update_loss_alpha(
            self, new_action_log_probs
    ):
        loss_alpha = (-self.log_alpha.exp() * (
                new_action_log_probs.detach() + self._target_entropy
        )).mean()

        logs = {
            'Alpha': self.log_alpha.exp().detach(),
            'LossAlpha': loss_alpha.detach()
        }
        return loss_alpha, logs

    def _clip_actions(self, actions):
        epsilon = 1e-6
        lower = torch.from_numpy(self._env_spec.action_space.low).to(self.device) + epsilon
        upper = torch.from_numpy(self._env_spec.action_space.high).to(self.device) - epsilon
    
        clip_up = (actions > upper).float()
        clip_down = (actions < lower).float()
        with torch.no_grad():
            clip = ((upper - actions) * clip_up + (lower - actions) * clip_down)
        return actions + clip