import copy
import torch
import numpy as np
from matplotlib import cm
from networks.poolers.poolers import get_pooler_network
from networks.hypernetwork.hypernetwork import HyperNetwork, PairedNetwork
from torch.optim import AdamW
from eval_utils.eval_utils import get_option_colors
from networks.utils.mlp_builder import mlp_builder

class METRA(torch.nn.Module):
    def __init__(
            self,
            *,
            obs_length,
            pooler_config, traj_encoder_config,
            lr, wd,
            dual_lam,
            option_size,
            discrete,
            unit_length,
            device,
            dual_slack,
    ):
        super().__init__()
        self.device = device
        self.preset_options = None
        self.pooler = get_pooler_network(name = pooler_config.name, obs_length = obs_length, 
                                         skill_length = option_size, pooler_config = pooler_config.kwargs).to(self.device)
        
        self._init_trajectory_encoder(obs_dim = self.pooler.outp_dim, skill_dim = option_size, 
                                      net_hidden_sizes = traj_encoder_config.net.hidden_sizes,
                                      net_hidden_nonlinearity = traj_encoder_config.net.hidden_nonlinearity,
                                      hypernet_param_dim = traj_encoder_config.hypernet.parameterizer_dim,
                                      hypernet_hidden_sizes = traj_encoder_config.hypernet.hidden_sizes,
                                      hypernet_hidden_act = traj_encoder_config.hypernet.hidden_nonlinearity,
                                      hypernet_compressed_dim = traj_encoder_config.hypernet.compressed_dim,
                                      hypernet_type = traj_encoder_config.hypernet.type)
        self.log_dual_lam = torch.nn.Parameter(data = torch.log(torch.Tensor([dual_lam])).to(device), 
                                               requires_grad = True)

        self.option_size = option_size

        self.discrete = discrete
        self.unit_length = unit_length
        self.dual_slack = dual_slack
        self.optimizer = AdamW(params = self.parameters(), lr = lr, weight_decay = wd)

    def _init_trajectory_encoder(self, obs_dim, skill_dim, net_hidden_sizes, net_hidden_nonlinearity, 
                                 hypernet_param_dim, hypernet_hidden_act, hypernet_hidden_sizes, hypernet_compressed_dim,
                                 hypernet_type):
        self.enc, self.dec = None, None
        if (hypernet_param_dim is None) or (hypernet_hidden_sizes is None) or (hypernet_compressed_dim is None):
            self._traj_encoder = mlp_builder(in_dim = obs_dim, 
                                             net_architecture = net_hidden_sizes,
                                             out_dim = skill_dim,
                                             nonlinearity_name = net_hidden_nonlinearity).to(self.device)
            self._use_hyper_net = False
        elif hypernet_type in ['classic', 'stupid']:
            if hypernet_type == 'classic':
                self._traj_encoder = HyperNetwork(parameterizer_dim = hypernet_param_dim, 
                                                  net_in_dim = obs_dim, net_out_dim = skill_dim,
                                                  hypernet_arch = hypernet_hidden_sizes,
                                                  hypernet_act = hypernet_hidden_act,
                                                  compressed_dim = hypernet_compressed_dim,
                                                  net_arch = net_hidden_sizes, 
                                                  net_act = net_hidden_nonlinearity).to(self.device)
            elif hypernet_type == 'stupid':
                self._traj_encoder = PairedNetwork(net_in_dim = obs_dim, net_out_dim = skill_dim, 
                                                   net_arch = net_hidden_sizes, 
                                                   net_act = net_hidden_nonlinearity).to(self.device)
            def enc(x):
                return x[..., :-2]
            
            def dec(compressed_x):
                shape_without_features = compressed_x.shape[:-1]
                return torch.cat([compressed_x, torch.zeros(size = shape_without_features + (2,)).to(self.device)], 
                                 axis = -1)
            
            self.enc, self.dec, self._use_hyper_net = enc, dec, True
            
    def call_traj_encoder(self, object_representation, observations, obj_idxs):
        batch_size = obj_idxs.shape[0]
        if self._use_hyper_net:
            parameter = self.enc(observations[torch.arange(batch_size), obj_idxs]).detach()
            return self._traj_encoder(object_representation, parameter)
        
        return self._traj_encoder(object_representation)

    def sample_options_and_obj_idxs(self, batch_size, traj_len, skills_per_traj, n_objects):
        assert traj_len % skills_per_traj == 0, 'Maximal length of trajectory must be divisible by skills per trajectory'
        if self.discrete:
            options = np.eye(self.option_size)[np.random.randint(0, self.option_size, size = (batch_size, skills_per_traj))]
        else:
            options = np.random.randn(batch_size, skills_per_traj, self.option_size).astype(np.float32)
            if self.unit_length:
                options /= np.linalg.norm(options, axis = -1, keepdims = True)
        options = np.repeat(options, traj_len // skills_per_traj, axis = 1)
        obj_idxs = np.random.randint(low = 0, high = n_objects, size = (batch_size, 1))
        obj_idxs = np.repeat(obj_idxs, traj_len, axis = 1)

        return options, obj_idxs
    
    def train_components(self, observations, next_observations, options, obj_idxs):
        cur_obj_repr = self.fetch_single_vector_representation(observations, obj_idxs)
        next_obj_repr = self.fetch_single_vector_representation(next_observations, obj_idxs)
        te_logs, loss_te, cst_penalty = self._update_loss_te(cur_obj_repr = cur_obj_repr, next_obj_repr = next_obj_repr, 
                                                             options = options,
                                                             observations = observations, next_observations = next_observations, 
                                                             obj_idxs = obj_idxs)
        dual_logs, loss_dual_lam = self._update_loss_dual_lam(cst_penalty)
        consistency_loss = self._update_loss_static(observations, next_observations, obj_idxs)
        
        rew_logs, cur_z, next_z, rewards = self._update_rewards(cur_obj_repr = cur_obj_repr, next_obj_repr = next_obj_repr, 
                                                                options = options,
                                                                observations = observations, next_observations = next_observations,
                                                                obj_idxs = obj_idxs)
        loss = loss_te + loss_dual_lam + consistency_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logs = {**te_logs, **dual_logs, **rew_logs}
        return logs, rewards
    
    def fetch_single_vector_representation(self, observations, obj_idxs):
        return self.pooler(observations, obj_idxs)
    
    def fetch_encoder_representation(self, observations, obj_idxs):
        with torch.no_grad():
            obj_repr = self.pooler(observations, obj_idxs)
            mean = self.call_traj_encoder(object_representation = obj_repr, observations = observations,
                                          obj_idxs = obj_idxs)
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
            rewards = self._update_rewards(cur_obj_repr, next_obj_repr, options,
                                           observations, next_observations, obj_idxs)[3]
        return rewards.reshape((traj_qty, traj_length) + rewards.shape[2:]).cpu().numpy()
    
    def _update_rewards(self, cur_obj_repr, next_obj_repr, options,
                        observations, next_observations, obj_idxs):
        logs = {}
        cur_z = self.call_traj_encoder(object_representation = cur_obj_repr, observations = observations,
                                       obj_idxs = obj_idxs)
        next_z = self.call_traj_encoder(object_representation = next_obj_repr, observations = next_observations,
                                        obj_idxs = obj_idxs)
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
    
    def _update_loss_te(self, cur_obj_repr, next_obj_repr, options,
                        observations, next_observations, obj_idxs):
        logs, cur_z, next_z, rewards = self._update_rewards(cur_obj_repr = cur_obj_repr, 
                                                            next_obj_repr = next_obj_repr,
                                                            options = options, 
                                                            observations = observations,
                                                            next_observations = next_observations,
                                                            obj_idxs = obj_idxs)
        
        dual_lam = self.log_dual_lam.exp()
        # Define temporal distance as distance between adjacent steps
        cst_dist = torch.ones(cur_obj_repr.shape[0]).to(cur_obj_repr.device)
        cst_penalty = cst_dist - torch.square(next_z - cur_z).mean(dim=1)
        cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)

        te_obj = rewards + dual_lam.detach() * cst_penalty
        logs.update({
            'DualCstPenalty': cst_penalty.mean(),
        })
        loss_te = -te_obj.mean()
        logs.update({
            'TeObjMean': te_obj.mean().detach(),
            'LossTe': loss_te.detach(),
        })
        return logs, loss_te, cst_penalty
            
    def _update_loss_dual_lam(self, cst_penalty):
        logs = {}
        dual_lam = self.log_dual_lam.exp()
        loss_dual_lam = self.log_dual_lam * (cst_penalty.detach()).mean()
        logs.update({
            'DualLam': dual_lam.detach(),
            'LossDualLam': loss_dual_lam.detach(),
        })
        return logs, loss_dual_lam
    
    def _update_loss_static(self, cur_obs, next_obs, obj_idxs):
        batch_size, *_ = obj_idxs.shape
        cur_obj, next_obj = cur_obs[torch.arange(batch_size), obj_idxs], next_obs[torch.arange(batch_size), obj_idxs]
        enc_cur_obj, enc_next_obj = self.enc(cur_obj), self.enc(next_obj)
        dec_cur_obj, dec_next_obj = self.dec(enc_cur_obj), self.dec(enc_next_obj)
        diff = torch.mean((cur_obj - dec_cur_obj)**2, axis=-1) + torch.mean((next_obj - dec_next_obj)**2, axis=-1) +\
            torch.mean(2*(enc_cur_obj - enc_next_obj)**2, axis=-1)
        return torch.mean(diff, dim = 0)

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
                video_options = video_options.repeat(2, axis = 0) # Num video repeats???
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