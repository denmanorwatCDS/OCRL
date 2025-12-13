import copy
import torch
import numpy as np
from matplotlib import cm
from networks.poolers.poolers import get_pooler_network
from networks.distributions.distributions import get_distribution_network
from networks.utils.parameter import ParameterModule
from networks.utils.mlp import MultiHeadedMLPModule
from torch.optim import AdamW
from eval_utils.eval_utils import get_option_colors

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
        self.preset_options = None
        self.pooler, self.is_pooler_trainable = get_pooler_network(name = pooler_config.name, obs_length = obs_length, 
                                                                   skill_length = option_size, 
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

    def sample_options_and_obj_idxs(self, batch_size, traj_len, skills_per_traj, n_objects):
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

        return options, obj_idxs
    
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
            optimizer_keys=['traj_encoder'] + (['pooler'] if self.is_pooler_trainable else []),
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
    
    def fetch_encoder_representation(self, observations, obj_idxs):
        with torch.no_grad():
            obj_repr = self.pooler(observations, obj_idxs)
            mean = self.traj_encoder(obj_repr)[0]
        mean = mean.cpu().numpy()
        std = np.ones_like(mean)
        samples = mean
        return mean, std, samples
    
    def calculate_rewards(self, observations, next_observations, options, obj_idxs):
        traj_qty, traj_length = observations.shape[:2]
        observations = torch.from_numpy(observations.reshape((-1,) + observations.shape[2:])).to(self.device)
        next_observations = torch.from_numpy(next_observations.reshape((-1,) + next_observations.shape[2:])).to(self.device)
        obj_idxs = torch.from_numpy(obj_idxs.reshape(-1)).to(self.device)
        options = torch.from_numpy(options.reshape((-1,) + options.shape[2:])).to(self.device)
        with torch.no_grad():
            cur_obj_repr = self.fetch_single_vector_representation(observations, obj_idxs)
            next_obj_repr = self.fetch_single_vector_representation(next_observations, obj_idxs)
            rewards = self._update_rewards(cur_obj_repr, next_obj_repr, options)[3]
        return rewards.reshape((traj_qty, traj_length) + rewards.shape[2:]).cpu().numpy()
    
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

    def sample_eval_options(self, num_random_trajectories, traj_length):
        random_options, option_colors = None, None
        if self.discrete:
            random_options, option_colors = self._sample_discrete_options(num_random_trajectories)
        else:
            random_options, option_colors = self._sample_continuous_options(num_random_trajectories)
        random_options = np.expand_dims(random_options, axis = 1)
        random_options = np.repeat(random_options, traj_length, axis = 1)
        return random_options, option_colors

    def _sample_discrete_options(self, num_random_trajectories):
        eye_options = np.eye(self.option_size)
        random_options = []
        colors = []
        for i in range(self.option_size):
            num_trajs_per_option = num_random_trajectories // self.option_size + (i < num_random_trajectories % self.option_size)
            for _ in range(num_trajs_per_option):
                random_options.append(eye_options[i])
                colors.append(i)
        random_options = np.array(random_options)
        colors = np.array(colors)
        num_evals = len(random_options)
        cmap = 'tab10' if self.option_size <= 10 else 'tab20'
        random_option_colors = []
        for i in range(num_evals):
            random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
        random_option_colors = np.array(random_option_colors)
        return random_options, random_option_colors

    def _sample_continuous_options(self, num_random_trajectories):
        random_options = np.random.randn(num_random_trajectories, self.option_size).astype(np.float32)
        if self.unit_length:
            random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
        random_option_colors = get_option_colors(random_options * 4)
        return random_options, random_option_colors
    
    def sample_fixated_options(self, traj_length):
        if self.preset_options is None:
            video_options = None
            if self.discrete:
                video_options = np.eye(self.option_size)
                video_options = video_options.repeat(2, axis=0) # Num video repeats???
            else:
                if self.option_size == 2:
                    radius = 1. if self.unit_length else 1.5
                    video_options = []
                    for angle in [3, 2, 1, 4]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options.append([0, 0])
                    for angle in [0, 5, 6, 7]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options = np.array(video_options)
                else:
                    video_options = np.random.randn(8, self.option_size)
                    if self.unit_length:
                        video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
                video_options = video_options.repeat(2, axis=0).astype(np.float32)
            video_options = np.expand_dims(video_options, axis = 1)
            video_options = np.repeat(video_options, traj_length, axis = 1)
            self.preset_options = video_options
        return copy.deepcopy(self.preset_options)