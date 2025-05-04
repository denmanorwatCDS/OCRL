import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.nn.utils import clip_grad_norm_
from torch.distributions import kl_divergence
from statistics import mean
from copy import deepcopy


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO(nn.Module):
    def __init__(self, 
                 vf,
                 clip_coef,
                 clip_vloss,
                 ent_coef,
                 vf_coef,
                 normalize_advantage,
                 max_grad_norm,
                 target_kl,
                 option_policy,
                 optimizers,
                 device
                 ):
        super().__init__()
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.normalize_advantage = normalize_advantage
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self._optimizers = optimizers
        self.device = device
        self.critic = vf.to(device)
        self.option_policy = option_policy.to(device)
        self.old_option_policy = deepcopy(option_policy)
        self.old_option_policy.requires_grad_ = False
        self.current_loss_coef = 1.
        self.mean_clipfracs = []

    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }
    
    @property
    def on_policy(self):
        return True
    
    def eval(self):
        self.option_policy.eval(), self.critic.eval()

    def train(self):
        self.option_policy.train(), self.critic.train()

    def optimize_op(self, batch):
        logs = self.update_loss_act(batch)
        if logs is None:
            return logs
        logs.update(self.update_loss_vf(batch))
        self._gradient_descent(logs['value_loss'] + logs['actor_loss'], ['option_policy', 'vf'])
        return logs

    def _gradient_descent(self, loss, optimizer_keys):
        for key in optimizer_keys:
            self._optimizers[key].zero_grad()
        loss.backward()
        clip_grad_norm_(self.option_policy.parameters(), self.max_grad_norm)
        clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        for key in optimizer_keys:
            self._optimizers[key].step()

    def update_loss_act(self, batch):
        # TODO uncomment lines
        obs = batch['obs']
        #option, obj_idx = batch['options'], batch['obj_idxs']
        actions, pre_tanh_actions, old_logprobs = batch['actions'], batch['pre_tanh_actions'], batch['log_probs']
        opt_input = {'obs': obs}
        #opt_input.update({'options': option, 'obj_idxs': obj_idx})
        new_logprobs, entropy, info = self.option_policy.get_logprob_and_entropy(opt_input, actions, pre_tanh_actions)
        log_ratio = new_logprobs - old_logprobs
        ratio = log_ratio.exp()

        with torch.no_grad():
            old_approx_kl = -(log_ratio).mean()
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clipfracs = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
            self.mean_clipfracs.append(clipfracs)
            if self.target_kl is not None and approx_kl > self.target_kl:
                return None
        
        advantages = batch['advantages']
        mean_adv, std_adv = advantages.mean(), advantages.std()
        if self.normalize_advantage:
            advantages = (advantages - mean_adv) / (std_adv + 1e-8)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        entropy_loss = entropy.mean()
        actor_loss = (pg_loss - self.ent_coef * entropy_loss) * self.current_loss_coef
        return {'entropy_loss': entropy_loss.detach().cpu(),
                'policy_gradient_loss': pg_loss.detach().cpu(),
                'clip_fractions': clipfracs,
                'old_approx_kl': old_approx_kl.detach().cpu(),
                'approx_kl': approx_kl.detach().cpu(),
                'actor_loss': actor_loss,
                'mean_advantage': mean_adv,
                'std_advantage': std_adv,
                'current loss coefficient': self.current_loss_coef,
                'mean_policy_std': torch.mean(info['normal_std'])}
    
    def update_loss_vf(self, batch):
        # TODO uncomment lines
        obs = batch['obs']
        #option, obj_idx = batch['options'], batch['obj_idxs']
        opt_input = {'obs': obs}
        #opt_input.update({'options': option, 'obj_idxs': obj_idx})
        new_values = self.critic(opt_input)
        returns, values = batch['returns'], batch['values']
        if self.clip_vloss:
            vf_loss_unclipped = (new_values - returns)**2
            new_values_clipped = values + torch.clamp(new_values - values, -self.clip_coef, self.clip_coef)
            vf_loss_clipped = (new_values_clipped - returns)**2
            vf_loss_max = torch.max(vf_loss_unclipped, vf_loss_clipped)
            v_loss = self.vf_coef * vf_loss_max.mean()
        else:
            v_loss = self.vf_coef * ((new_values - returns)**2).mean()
        
        return {'value_loss': 0.5 * v_loss, 
                'return_mean': returns.mean(),
                'returns_std': returns.std()}
    
    def update_old_option_policy(self):
        self.old_option_policy = deepcopy(self.option_policy)
        self.old_option_policy.requires_grad_ = False
    
    def reset_optimizers(self):
        clipfracs = mean(self.mean_clipfracs)
        self.mean_clipfracs = []
        if clipfracs > 0.45:
            self.current_loss_coef * 0.8
        elif clipfracs < 0.05:
            self.current_loss_coef /= 0.8
        for key in self._optimizers.keys():
            state_param_dict = self._optimizers[key].state_dict()
            state_param_dict = {'state': {}, 'param_groups': state_param_dict['param_groups']}
            self._optimizers[key].load_state_dict(state_param_dict)