import copy

import torch
import numpy as np
from optimizers.optimizer_wrapper import OptimizerGroupWrapper

def get_torch_concat_obs(obs, option, dim=1):
    concat_obs = torch.cat([obs] + [option], dim=dim)
    return concat_obs

class METRA():
    def __init__(
            self,
            *,
            traj_encoder,
            dist_predictor,
            dual_lam,
            optimizers,
            dim_option,
            discrete,
            unit_length,
            device,

            dual_reg,
            dual_slack,
            dual_dist,
    ):
        self.device = device
        self.traj_encoder = traj_encoder.to(self.device)

        self.dual_lam = dual_lam.to(self.device)
        self.param_modules = {
            'traj_encoder': self.traj_encoder,
            'dual_lam': self.dual_lam,
        }

        if dist_predictor is not None:
            self.dist_predictor = dist_predictor
            self.dist_predictor.to(self.device)
            self.param_modules['dist_predictor'] = self.dist_predictor

        self.dim_option = dim_option

        self._optimizers = optimizers
        self.discrete = discrete
        self.unit_length = unit_length
        self.traj_encoder.eval()

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist

    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def _get_train_trajectories_kwargs(self, batch_size, traj_len, skills_per_traj, n_objects):
        assert traj_len % skills_per_traj == 0, 'Maximal length of trajectory must be divisible by skills per trajectory'
        if self.discrete:
            options = np.eye(self.dim_option)[np.random.randint(0, self.dim_option, size = (batch_size, skills_per_traj))]
        else:
            options = np.random.randn(batch_size, skills_per_traj, self.dim_option).astype(np.float32)
            if self.unit_length:
                options /= np.linalg.norm(options, axis=-1, keepdims=True)
        options = np.repeat(options, traj_len // skills_per_traj, axis=1)
        obj_idxs = np.random.randint(low = 0, high = n_objects, size = (batch_size, 1))
        obj_idxs = np.repeat(obj_idxs, traj_len, axis=1)
        
        extras = []
        for i in range(batch_size):
            extras.append([])
            for traj_idx in range(traj_len):
                extras[-1].append({'options': options[i, traj_idx], 'obj_idxs': obj_idxs[i, traj_idx]})
        return extras

    def train_components(self, batch):
        logs = {}
        modified_batch = copy.deepcopy(batch)
        logs.update(self._optimize_te(modified_batch))
        logs.update(self._update_rewards(modified_batch))

        return logs, modified_batch

    def _optimize_te(self, batch):
        logs = self._update_loss_te(batch)

        self._gradient_descent(
            logs['LossTe'],
            optimizer_keys=['traj_encoder'],
        )

        if self.dual_reg:
            logs.update(self._update_loss_dual_lam(batch))
            self._gradient_descent(
                logs['LossDualLam'],
                optimizer_keys=['dual_lam'],
            )
            if self.dual_dist == 's2_from_s':
                self._gradient_descent(
                    logs['LossDp'],
                    optimizer_keys=['dist_predictor'],
                )
        return logs

    def _update_rewards(self, batch):
        logs = {}
        obs = {'obs': batch['obs'], 'obj_idxs': batch['obj_idxs']}
        next_obs = {'obs': batch['next_obs'], 'obj_idxs': batch['obj_idxs']}

        cur_z = self.traj_encoder(obs).mean
        next_z = self.traj_encoder(next_obs).mean
        target_z = next_z - cur_z

        if self.discrete:
            masks = (batch['options'] - batch['options'].mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
            rewards = (target_z * masks).sum(dim=1)
        else:
            inner = (target_z * batch['options']).sum(dim=1)
            rewards = inner

        # For dual objectives
        batch.update({
            'cur_z': cur_z,
            'next_z': next_z,
        })

        logs.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })

        batch['rewards'] = rewards
        return logs

    def _update_loss_te(self, batch):
        logs = {}
        logs.update(self._update_rewards(batch))
        rewards = batch['rewards']

        obs = {'obs': batch['obs']}
        next_obs = {'obs': batch['next_obs']}

        if self.dual_dist == 's2_from_s':
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            logs.update({
                'LossDp': loss_dp,
            })

        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs['obs']
            y = next_obs['obs']
            phi_x = batch['cur_z']
            phi_y = batch['next_z']

            if self.dual_dist == 'l2':
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.dual_dist == 'one':
                cst_dist = torch.ones(x.shape[0]).to(x.device)
            elif self.dual_dist == 's2_from_s':
                s2_dist = self.dist_predictor(obs)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                scaling_factor = 1. / s2_dist_std
                geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(torch.square((y - x) - s2_dist_mean) * normalized_scaling_factor, dim=1)

                # TODO it is 2-dimensional tensor. Is usage of mean across all dimensions is justified?
                logs.update({
                    'ScalingFactor': scaling_factor.mean(),
                    'NormalizedScalingFactor': normalized_scaling_factor.mean(),
                })
            else:
                raise NotImplementedError

            cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
            te_obj = rewards + dual_lam.detach() * cst_penalty

            batch.update({
                'cst_penalty': cst_penalty
            })
            logs.update({
                'DualCstPenalty': cst_penalty.mean(),
            })
        else:
            te_obj = rewards

        loss_te = -te_obj.mean()

        logs.update({
            'TeObjMean': te_obj.mean(),
            'LossTe': loss_te,
        })
        return logs

    def _update_loss_dual_lam(self, batch):
        logs = {}
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (batch['cst_penalty'].detach()).mean()

        logs.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })
        return logs

    def _gradient_descent(self, loss, optimizer_keys):
        for key in optimizer_keys:
            self._optimizers[key].zero_grad()
        loss.backward()
        for key in optimizer_keys:
            self._optimizers[key].step()