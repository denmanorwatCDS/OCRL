import copy
import torch
import torch.nn.functional as F
import numpy as np
from RL.policies.policy_v2 import Policy
from networks.poolers.poolers import get_pooler_network
from torch.optim import AdamW
from itertools import chain
from networks.utils.parameter import ParameterModule

class SAC(Policy):
    def __init__(self,
                 name,
                 obs_length, task_length, action_length,
                 actor_config, critic_config, pooler_config,
                 actor_lr, critic_lr, pooler_lr, log_alpha_lr,
                 actor_wd, critic_wd, pooler_wd, log_alpha_wd,
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
        super(Policy, self).__init__()
        self.pooler, self.is_pooler_trainable = get_pooler_network(name = pooler_config.name, obs_length = obs_length, 
                                                                   pooler_config = pooler_config.kwargs)
        self.pooler.to(device)
        super().__init__(name = name, device = device, obs_length = self.pooler.outp_dim, task_length = task_length, action_length = action_length,
                         account_for_action = True, actor_config = actor_config, critic_config = critic_config,
                         clip_action = clip_action, force_use_mode_actions = force_use_mode_actions)
        self.log_alpha = ParameterModule(torch.Tensor([np.log(alpha)])).to(device)
        self.build_optimizers(actor_lr = actor_lr, critic_lr = critic_lr, pooler_lr = pooler_lr, 
                              log_alpha_lr = log_alpha_lr, actor_wd = actor_wd, critic_wd = critic_wd, 
                              pooler_wd = pooler_wd, log_alpha_wd = log_alpha_wd)
        self.discount = discount
        self.device = device

        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        self.tau = tau
        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(env_spec.action_space.shape).item() / 2. * target_coef

    def build_optimizers(self, actor_lr, critic_lr, pooler_lr, log_alpha_lr,
                               actor_wd, critic_wd, pooler_wd, log_alpha_wd):
        self._optimizers = {'actor': AdamW(params = self.actor.parameters(), lr = actor_lr, weight_decay = actor_wd),
                            'critic': AdamW(params = chain(self.critic1.parameters(), self.critic2.parameters()), 
                                            lr = critic_lr, weight_decay = critic_wd),
                            'log_alpha': AdamW(params = self.log_alpha.parameters(), 
                                               lr = log_alpha_lr, weight_decay = log_alpha_wd)}
        if self.is_pooler_trainable:
            self._optimizers['pooler'] = AdamW(params = self.pooler.parameters(), lr = pooler_lr, weight_decay = pooler_wd)

    @property
    def policy(self):
        return {
            'option_policy': self.actor,
        }
    
    @property
    def on_policy(self):
        return False
    
    def eval(self):
        self.actor.eval(), self.critic1.eval(), self.critic2.eval(), self.target_critic1.eval(), self.target_critic2.eval()

    def train(self):
        self.actor.train(), self.critic1.train(), self.critic2.train(), self.target_critic1.train(), self.target_critic2.train()

    def optimize_op(self, observations, next_observations, obj_idxs, options, actions, dones, rewards):
        logs = {}
        cur_features = self.pooler(observations, obj_idxs)
        next_features = self.pooler(next_observations, obj_idxs)
        loss_qf, qf_logs = self._update_loss_qf(
            cur_features = cur_features,
            options = options,
            actions = actions,
            next_features = next_features,
            dones = dones,
            rewards = rewards * self._reward_scale_factor
        )
        self._gradient_descent(
            loss_qf,
            optimizer_keys=['critic'] + (['pooler'] if self.is_pooler_trainable else []),
            retain_graph=(True if self.is_pooler_trainable else False)
        )
        
        new_action_log_probs, sacp_loss, sacp_logs = self._update_loss_sacp(cur_features = cur_features,
                                                                            options = options)
        self._gradient_descent(
            sacp_loss,
            optimizer_keys=['actor'] + (['pooler'] if self.is_pooler_trainable else []),
        )

        loss_alpha, alpha_logs = self._update_loss_alpha(new_action_log_probs)
        self._gradient_descent(
            loss_alpha,
            optimizer_keys=['log_alpha'],
        )
        logs.update({**qf_logs, **sacp_logs, **alpha_logs})
        if self.is_pooler_trainable:
            """
            logs.update({'Skill token mean': torch.mean(self.pooler.skill_token.detach().cpu()),
                         'Skill token std': torch.std(self.pooler.skill_token.detach().cpu()),
                         'Actor token mean': torch.mean(self.pooler.act_token.detach().cpu()),
                         'Actor token std': torch.std(self.pooler.act_token.detach().cpu()),
                         'Critic token mean': torch.mean(self.pooler.crit_token.detach().cpu()),
                         'Critic token std': torch.std(self.pooler.crit_token.detach().cpu())})
            """
        self._update_targets()
        return logs
    
    def inference_value(self, observations, actions, options, obj_idxs):
        with torch.no_grad():
            batch_length, horizon_length, obj_length = observations.shape[:3]
            observation_shape = observations.shape[3:]
            observations = torch.from_numpy(observations.reshape((batch_length * horizon_length, obj_length, *observation_shape))).to(self.device)
            actions = torch.from_numpy(actions.reshape((batch_length * horizon_length, -1))).to(self.device)
            options = torch.from_numpy(options.reshape((batch_length * horizon_length, -1))).to(self.device)
            obj_idxs = torch.from_numpy(obj_idxs.reshape((batch_length * horizon_length))).to(self.device)
            cur_features = self.pooler(observations, obj_idxs)
            values = torch.min(
                self.target_critic1(cur_features, options, actions).flatten(),
                self.target_critic2(cur_features, options, actions).flatten(),
            )
        return values.reshape((batch_length, horizon_length)).detach().cpu().numpy()

    def _gradient_descent(self, loss, optimizer_keys, retain_graph = False):
        for key in optimizer_keys:
            self._optimizers[key].zero_grad()
        loss.backward(retain_graph = retain_graph)
        for key in optimizer_keys:
            self._optimizers[key].step()

    def _update_targets(self):
        target_qfs = [self.target_critic1, self.target_critic2]
        qfs = [self.critic1, self.critic2]
        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                                   param.data * self.tau)
                
    def _update_loss_qf(self,
        cur_features,
        options,
        actions,
        next_features,
        dones,
        rewards,
    ):
        with torch.no_grad():
            alpha = self.log_alpha.param.exp()
        q1_pred = self.critic1(cur_features, options, actions).flatten()
        q2_pred = self.critic2(cur_features, options, actions).flatten()
        
        next_action_dists, *_ = self.forward(next_features, options)
        if hasattr(next_action_dists, 'rsample_with_pre_tanh_value'):
            new_next_actions_pre_tanh, new_next_actions = next_action_dists.rsample_with_pre_tanh_value()
            new_next_action_log_probs = next_action_dists.log_prob(new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)
        else:
            new_next_actions = next_action_dists.rsample()
            new_next_actions = self._clip_actions(new_next_actions)
            new_next_action_log_probs = next_action_dists.log_prob(new_next_actions)

        target_q_values = torch.min(
            self.target_critic1(next_features, options, new_next_actions).flatten(),
            self.target_critic2(next_features, options, new_next_actions).flatten(),
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
            self, cur_features, options, 
    ):
        with torch.no_grad():
            alpha = self.log_alpha.param.exp()

        action_dists, *_ = self.forward(cur_features, options)
        if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
            new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
            new_action_log_probs = action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)
        else:
            new_actions = action_dists.rsample()
            new_actions = self._clip_actions(new_actions)
            new_action_log_probs = action_dists.log_prob(new_actions)
        min_q_values = torch.min(
            self.critic1(cur_features, options, new_actions).flatten(),
            self.critic2(cur_features, options, new_actions).flatten(),
        )

        loss_sacp = (alpha * new_action_log_probs - min_q_values).mean()

        return new_action_log_probs, loss_sacp, {
            'LossSacp': loss_sacp.detach(),
            'SacpNewActionLogProbMean': new_action_log_probs.mean()
        }

    def _update_loss_alpha(
            self, new_action_log_probs
    ):
        loss_alpha = (-self.log_alpha.param * (
                new_action_log_probs.detach() + self._target_entropy
        )).mean()

        logs = {
            'Alpha': self.log_alpha.param.exp().detach(),
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