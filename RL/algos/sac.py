import copy
import torch
import torch.nn.functional as F
import numpy as np

from optimizers.optimizer_wrapper import OptimizerGroupWrapper

def get_torch_concat_obs(obs, option, dim = 1):
    concat_obs = torch.cat([obs] + [option], dim = dim)
    return concat_obs

class SAC:
    def __init__(
        self,
        qf1,
        qf2,
        log_alpha,
        tau,
        scale_reward,
        env_spec,
        target_coef,
        option_policy,
        optimizers,
        device,
        discount):

        self.discount = discount
        self.device = device

        self.option_policy = option_policy.to(self.device)

        self._optimizers = optimizers

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.param_modules = {
            "qf1": self.qf1, "qf2": self.qf2, "log_alpha": self.log_alpha
            }
        
        self.tau = tau
        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(env_spec.action_space.shape).item() / 2. * target_coef

    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }
    
    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def optimize_op(self, modified_batch):

        logs = self._update_loss_qf(
            obs=modified_batch['obs'],
            actions=modified_batch['actions'],
            next_obs=modified_batch['next_obs'],
            dones=modified_batch['dones'],
            rewards=modified_batch['rewards'] * self._reward_scale_factor,
            policy=self.option_policy,
        )
        self._gradient_descent(
            logs['LossQf1'] + logs['LossQf2'],
            optimizer_keys=['qf'],
        )
        
        logs.update(self._update_loss_sacp(modified_batch, obs=modified_batch['obs'],))
        self._gradient_descent(
            logs['LossSacp'],
            optimizer_keys=['option_policy'],
        )

        logs.update(self._update_loss_alpha(modified_batch))
        self._gradient_descent(
            logs['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

        self.update_targets()
        return logs

    def _gradient_descent(self, loss, optimizer_keys):
        for key in optimizer_keys:
            self._optimizers[key].zero_grad()
        loss.backward()
        for key in optimizer_keys:
            self._optimizers[key].step()

    def update_targets(self):
        target_qfs = [self.target_qf1, self.target_qf2]
        qfs = [self.qf1, self.qf2]
        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                                   param.data * self.tau)
                
    def _update_loss_qf(self,
        obs,
        actions,
        next_obs,
        dones,
        rewards,
        policy,
    ):
        with torch.no_grad():
            alpha = self.log_alpha.param.exp()

        q1_pred = self.qf1(obs, actions).flatten()
        q2_pred = self.qf2(obs, actions).flatten()

        next_action_dists, *_ = policy(next_obs)
        if hasattr(next_action_dists, 'rsample_with_pre_tanh_value'):
            new_next_actions_pre_tanh, new_next_actions = next_action_dists.rsample_with_pre_tanh_value()
            new_next_action_log_probs = next_action_dists.log_prob(new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)
        else:
            new_next_actions = next_action_dists.rsample()
            new_next_actions = self._clip_actions(new_next_actions)
            new_next_action_log_probs = next_action_dists.log_prob(new_next_actions)

        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions).flatten(),
            self.target_qf2(next_obs, new_next_actions).flatten(),
        )
        target_q_values = target_q_values - alpha * new_next_action_log_probs
        target_q_values = target_q_values * self.discount

        with torch.no_grad():
            q_target = rewards + target_q_values * (1. - dones)

        # critic loss weight: 0.5
        loss_qf1 = F.mse_loss(q1_pred, q_target) * 0.5
        loss_qf2 = F.mse_loss(q2_pred, q_target) * 0.5

        return {
            'QTargetsMean': q_target.mean(),
            'QTdErrsMean': ((q_target - q1_pred).mean() + (q_target - q2_pred).mean()) / 2,
            'LossQf1': loss_qf1,
            'LossQf2': loss_qf2,
        }
        
    def _update_loss_sacp(
            self, batch, obs
    ):
        with torch.no_grad():
            alpha = self.log_alpha.param.exp()

        action_dists, *_ = self.option_policy(obs)
        if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
            new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
            new_action_log_probs = action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)
        else:
            new_actions = action_dists.rsample()
            new_actions = self._clip_actions(new_actions)
            new_action_log_probs = action_dists.log_prob(new_actions)

        min_q_values = torch.min(
            self.qf1(obs, new_actions).flatten(),
            self.qf2(obs, new_actions).flatten(),
        )

        loss_sacp = (alpha * new_action_log_probs - min_q_values).mean()

        batch.update({
            'new_action_log_probs': new_action_log_probs,
        })

        return {
            'SacpNewActionLogProbMean': new_action_log_probs.mean(),
            'LossSacp': loss_sacp,
        }

    def _update_loss_alpha(
            self, batch,
    ):
        loss_alpha = (-self.log_alpha.param * (
                batch['new_action_log_probs'].detach() + self._target_entropy
        )).mean()

        logs = {
            'Alpha': self.log_alpha.param.exp(),
            'LossAlpha': loss_alpha,
        }
        return logs

    def _clip_actions(self, actions):
        epsilon = 1e-6
        lower = torch.from_numpy(self._env_spec.action_space.low).to(self.device) + epsilon
        upper = torch.from_numpy(self._env_spec.action_space.high).to(self.device) - epsilon
    
        clip_up = (actions > upper).float()
        clip_down = (actions < lower).float()
        with torch.no_grad():
            clip = ((upper - actions) * clip_up + (lower - actions) * clip_down)
    
        return actions + clip