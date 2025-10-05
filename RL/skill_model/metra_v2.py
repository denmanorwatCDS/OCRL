import copy
import torch
import numpy as np
from networks.poolers.poolers import get_pooler_network
from networks.distributions.distributions import get_distribution_network
from networks.utils.parameter import ParameterModule
from networks.utils.mlp import MultiHeadedMLPModule
from torch.optim import AdamW

class METRA(torch.nn.Module):
    def __init__(
            self,
            *,
            obs_length,
            pooler_config, traj_encoder_config, dist_predictor_config,
            pooler_lr, traj_lr, dist_lr = None, dual_lam_lr,
            pooler_wd, traj_wd, dist_wd = None, dual_lam_wd,
            dist_predictor_name,
            dual_lam,
            option_size,
            discrete,
            unit_length,
            device,

            dual_reg,
            dual_slack,
    ):
        super().__init__()
        self.device = device
        self.pooler, self.is_pooler_trainable = get_pooler_network(name = pooler_config.name, obs_length = obs_length, 
                                                              pooler_config = pooler_config.kwargs)
        self.pooler.to(self.device)
        self.traj_encoder = MultiHeadedMLPModule(n_heads = 1, input_dim = self.pooler.outp_dim, output_dims = option_size,
                                                 **traj_encoder_config.mlp).to(self.device)
        self.log_dual_lam = ParameterModule(torch.Tensor([np.log(dual_lam)])).to(self.device)
        self.dist_predictor_name = dist_predictor_name

        if dist_predictor_name == 's2_from_s':
            self.dist_predictor = get_distribution_network(name = dist_predictor_config.name, obs_length = obs_length, task_length = 0, 
                                                           output_dim = 1, **dist_predictor_config.distribution)
            self.dist_predictor.to(self.device)
        self.build_optimizers(pooler_lr = pooler_lr, traj_lr = traj_lr, dist_lr = dist_lr, dual_lam_lr = dual_lam_lr,
                              pooler_wd = pooler_wd, traj_wd = traj_wd, dist_wd = dist_wd, dual_lam_wd = dual_lam_wd)

        self.option_size = option_size

        self.discrete = discrete
        self.unit_length = unit_length
        self.traj_encoder.eval()

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack

    def build_optimizers(self, pooler_lr, traj_lr, dist_lr, dual_lam_lr,
                               pooler_wd, traj_wd, dist_wd, dual_lam_wd):
        
        self._optimizers = {'traj_encoder': AdamW(params = self.traj_encoder.parameters(), lr = traj_lr,
                                                 weight_decay = traj_wd),
                            'dual_lam': AdamW(params = self.log_dual_lam.parameters(), lr = dual_lam_lr, 
                                              weight_decay = dual_lam_wd)}
        if self.dist_predictor_name == 's2_from_s':
            self._optimizers['dist_predictor'] = AdamW(params = self.dist_predictor.parameters(), 
                                                      lr = dist_lr, weight_decay = dist_wd)
            
        if self.is_pooler_trainable:
            self._optimizers['pooler'] = AdamW(params = self.pooler.parameters(), lr = pooler_lr, weight_decay = pooler_wd)

    def _get_train_trajectories_kwargs(self, batch_size, traj_len, skills_per_traj, n_objects):
        assert traj_len % skills_per_traj == 0, 'Maximal length of trajectory must be divisible by skills per trajectory'
        if self.discrete:
            options = np.eye(self.option_size)[np.random.randint(0, self.option_size, size = (batch_size, skills_per_traj))]
        else:
            options = np.random.randn(batch_size, skills_per_traj, self.option_size).astype(np.float32)
            if self.unit_length:
                options /= np.linalg.norm(options, axis = -1, keepdims = True)
        options = np.repeat(options, traj_len // skills_per_traj, axis=1)
        obj_idxs = np.random.randint(low = 0, high = n_objects, size = (batch_size, 1))
        obj_idxs = np.repeat(obj_idxs, traj_len, axis=1)

        extras = []
        for i in range(batch_size):
            extras.append([])
            for traj_idx in range(traj_len):
                extras[-1].append({'options': options[i, traj_idx], 'obj_idxs': obj_idxs[i, traj_idx]})
        return extras
    
    def train_components(self, observations, next_observations, options, obj_idxs):
        logs = {}
        cur_obj_repr = self.fetch_single_vector_representation(observations, obj_idxs)
        next_obj_repr = self.fetch_single_vector_representation(next_observations, obj_idxs)
        logs.update(self._optimize_te(cur_obj_repr = cur_obj_repr, next_obj_repr = next_obj_repr, options = options))
        rew_logs, cur_z, next_z, rewards = self._update_rewards(cur_obj_repr, next_obj_repr, options = options)
        logs.update(rew_logs)
        return logs, rewards
    
    def _optimize_te(self, cur_obj_repr, next_obj_repr, options):
        te_logs, loss_te, cst_penalty = self._update_loss_te(cur_obj_repr, next_obj_repr, options)
        self._gradient_descent(
            loss_te,
            optimizer_keys=['traj_encoder'] + ['pooler'] if self.is_pooler_trainable else [],
        )
        if self.dual_reg:
            dual_logs, loss_dual_lam = self._update_loss_dual_lam(cst_penalty)
            self._gradient_descent(
                loss_dual_lam,
                optimizer_keys=['dual_lam'],
            )
            if self.dist_predictor_name == 's2_from_s':
                pass
                # self._gradient_descent(
                #    logs['LossDp'],
                #    optimizer_keys=['dist_predictor'],
                # )
        return {**te_logs, **dual_logs}
    
    def fetch_single_vector_representation(self, observations, obj_idxs):
        return self.pooler(observations, obj_idxs)
    
    def fetch_trajectory_encoder_representation(self, observations, obj_idxs):
        obj_repr = self.pooler(observations, obj_idxs)
        return self.traj_encoder(obj_repr)[0]
    
    def _update_rewards(self, cur_obj_repr, next_obj_repr, options):
        logs = {}
        cur_z = self.traj_encoder(cur_obj_repr)[0]
        next_z = self.traj_encoder(next_obj_repr)[0]
        target_z = next_z - cur_z
        if self.discrete:
            masks = (options - options.mean(dim=1, keepdim=True)) * self.option_size / (self.option_size - 1 if self.option_size != 1 else 1)
            rewards = (target_z * masks).sum(dim=1)
        else:
            inner = (target_z * options).sum(dim=1)
            rewards = inner
        # For dual objectives
        logs.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })
        return logs, cur_z, next_z, rewards
    
    def _update_loss_te(self, cur_obj_repr, next_obj_repr, options):
        logs, cur_z, next_z, rewards = self._update_rewards(cur_obj_repr = cur_obj_repr, 
                                                            next_obj_repr = next_obj_repr,
                                                            options = options)
        if self.dist_predictor_name == 's2_from_s':
            s2_dist = self.dist_predictor(cur_obj_repr)
            loss_dp = -s2_dist.log_prob(next_obj_repr - cur_obj_repr).mean()
            logs.update({
                'LossDp': loss_dp,
            })
        if self.dual_reg:
            dual_lam = self.log_dual_lam.param.exp()
            if self.dist_predictor_name == 'l2':
                cst_dist = torch.square(next_obj_repr - cur_obj_repr).mean(dim=1)
            elif self.dist_predictor_name == 'one':
                cst_dist = torch.ones(cur_obj_repr.shape[0]).to(cur_obj_repr.device)
            elif self.dist_predictor_name == 's2_from_s':
                # Was just obs. Check for errors
                s2_dist = self.dist_predictor(cur_obj_repr)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                scaling_factor = 1. / s2_dist_std
                geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(torch.square((next_obj_repr - cur_obj_repr) - s2_dist_mean) * normalized_scaling_factor, dim=1)
                # TODO it is 2-dimensional tensor. Is usage of mean across all dimensions is justified?
                logs.update({
                    'ScalingFactor': scaling_factor.mean(),
                    'NormalizedScalingFactor': normalized_scaling_factor.mean(),
                })
            else:
                raise NotImplementedError
            cst_penalty = cst_dist - torch.square(next_z - cur_z).mean(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
            te_obj = rewards + dual_lam.detach() * cst_penalty
            logs.update({
                'DualCstPenalty': cst_penalty.mean(),
            })
        else:
            te_obj = rewards
        loss_te = -te_obj.mean()
        logs.update({
            'TeObjMean': te_obj.mean().detach(),
            'LossTe': loss_te.detach(),
        })
        return logs, loss_te, cst_penalty
            
    def _update_loss_dual_lam(self, cst_penalty):
        logs = {}
        log_dual_lam = self.log_dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (cst_penalty.detach()).mean()
        logs.update({
            'DualLam': dual_lam.detach(),
            'LossDualLam': loss_dual_lam.detach(),
        })
        return logs, loss_dual_lam
    
    def _gradient_descent(self, loss, optimizer_keys):
        for key in optimizer_keys:
            self._optimizers[key].zero_grad()
        loss.backward()
        for key in optimizer_keys:
            self._optimizers[key].step()