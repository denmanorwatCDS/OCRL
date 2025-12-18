import torch
from torch import nn
import numpy as np
from RL.policies.policy_v2 import Actor

from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from networks.poolers.poolers import get_pooler_network

class PPO(nn.Module):
    def __init__(self,
                 name,
                 obs_length, task_length, action_length,
                 actor_config, critic_config, pooler_config,
                 actor_lr, critic_lr, pooler_lr,
                 actor_wd, critic_wd, pooler_wd,
                 clip_action=False,
                 force_use_mode_actions=False,
                 *,
                 clip_coef,
                 ent_coef,
                 vf_coef,
                 normalize_advantage,
                 max_grad_norm,
                 target_kl,
                 device
                 ):
        super().__init__()
        self.pooler, self.is_pooler_trainable = get_pooler_network(name = pooler_config.name, obs_length = obs_length, 
                                         pooler_config = pooler_config.kwargs)
        super().__init__(name = name, obs_length = self.pooler.outp_dim, task_length = task_length, 
                         action_length = action_length, account_for_action = False, actor_config = actor_config,
                         critic_config = critic_config, clip_action = clip_action,
                         force_use_mode_actions = force_use_mode_actions, device = device)
        self.build_optimizers(actor_lr, critic_lr, pooler_lr, actor_wd, critic_wd, pooler_wd)
        self.clip_coef = clip_coef
        self.rollback_alpha = -0.3
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.normalize_advantage = normalize_advantage
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = device
        self.mean_clipfracs = []

    def build_optimizers(self, actor_lr, critic_lr, pooler_lr,
                               actor_wd, critic_wd, pooler_wd,):
        self._optimizers = {'actor': AdamW(params = self.actor.parameters(), lr = actor_lr, weight_decay = actor_wd),
                            'critic': AdamW(params = self.critic1.parameters(), lr = critic_lr, weight_decay = critic_wd)}
        if self.is_pooler_trainable:
            self._optimizers['pooler'] = AdamW(params = self.pooler.parameters(), lr = pooler_lr, weight_decay = pooler_wd)

    @property
    def policy(self):
        return {
            'option_policy': self.actor,
        }
    
    @property
    def on_policy(self):
        return True
    
    def eval(self):
        self.actor.eval(), self.critic1.eval()

    def train(self):
        self.actor.train(), self.critic1.train()

    def optimize_op(self, observations, obj_idxs, options, actions, pre_tanh_actions, old_logprobs, advantages, returns):
        logs = {}
        act_vector, crit_vector = self.pooler(observations, obj_idxs)
        act_loss, act_logs = self.update_loss_act(act_vector, options, actions, pre_tanh_actions, old_logprobs, advantages)
        if act_logs is None:
            return logs
        v_loss, v_logs = self.update_loss_vf(crit_vector, options, returns)
        logs.update({**act_logs, **v_logs})
        self._gradient_descent(act_loss + v_loss, ['actor', 'critic'] + (['pooler'] if self.is_pooler_trainable else []))
        return logs

    def _gradient_descent(self, loss, optimizer_keys):
        for key in optimizer_keys:
            self._optimizers[key].zero_grad()
        loss.backward()
        clip_grad_norm_(list(self.actor.parameters()) + list(self.critic1.parameters()), self.max_grad_norm)
        for key in optimizer_keys:
            self._optimizers[key].step()

    def update_loss_act(self, actor_vector, options, actions, pre_tanh_actions, old_logprobs, advantages):
        new_logprobs, entropy, info = self.get_logprob_and_entropy(actor_vector, options, actions, pre_tanh_actions)
        log_ratio = new_logprobs - old_logprobs
        ratio = log_ratio.exp()

        with torch.no_grad():
            old_approx_kl = -(log_ratio).mean()
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clipfracs = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
            self.mean_clipfracs.append(clipfracs)
            if self.target_kl is not None and approx_kl > self.target_kl:
                return None
        
        mean_adv, std_adv = advantages.mean(), advantages.std()
        if self.normalize_advantage:
            advantages = (advantages - mean_adv) / (std_adv + 1e-8)
        clips = torch.where(advantages >= 0,
                            torch.where(ratio <= 1 + self.clip_coef, 
                                        ratio, 
                                        self.rollback_alpha * ratio + (1 - self.rollback_alpha) * (1 + self.clip_coef)),
                            torch.where(ratio >= 1 - self.clip_coef,
                                        ratio,
                                        self.rollback_alpha * ratio + (1 - self.rollback_alpha) * (1 - self.clip_coef)))
        pg_loss = clips * advantages
        pg_loss = -pg_loss.mean()

        entropy_loss = entropy.mean()
        actor_loss = (pg_loss - self.ent_coef * entropy_loss)
        return actor_loss, {'entropy_loss': entropy_loss.detach().cpu(),
                'policy_gradient_loss': pg_loss.detach().cpu(),
                'clip_fractions': clipfracs,
                'old_approx_kl': old_approx_kl.detach().cpu(),
                'approx_kl': approx_kl.detach().cpu(),
                'actor_loss': actor_loss,
                'mean_advantage': mean_adv,
                'std_advantage': std_adv,
                'mean_policy_std': torch.mean(info['normal_std'])}
    
    def update_loss_vf(self, critic_vector, options, returns):
        # TODO uncomment lines
        new_values = torch.squeeze(self.critic1(critic_vector, options))
        v_loss = 0.5 * ((new_values - returns)**2).mean() * self.vf_coef
        
        return v_loss, {'value_loss': v_loss/(0.5 * self.vf_coef), 
                'return_mean': returns.mean(),
                'returns_std': returns.std()}
    
    def get_critic_value(self, observations, options, obj_idxs):
        traj_qty, traj_length = observations.shape[:2]
        observations = torch.from_numpy(observations.reshape((-1,) + observations.shape[2:])).to(self.device)
        obj_idxs = torch.from_numpy(obj_idxs.reshape(-1)).to(self.device)
        options = torch.from_numpy(options.reshape((-1,) + options.shape[2:])).to(self.device)
        with torch.no_grad():
            _, critic_vector = self.pooler(observations, obj_idxs)
            values = self.critic1(critic_vector, options)
            return values.reshape((traj_qty, traj_length) + values.shape[2:]).cpu().numpy().astype(np.float32)
