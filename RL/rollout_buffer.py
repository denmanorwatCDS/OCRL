import torch
from copy import deepcopy

class OCRolloutBuffer():
    def __init__(self, obs_shape, action_shape, gamma, gae_lambda, device, seed, num_parallel_envs,
                 batch_size, rollout_max_epochs, ppg_max_epochs, use_future, steps_into_future,
                 random_dataset, dataset_preprocessor, 
                 augmenter = None, parallel_processes = 10, memory_size = 50_000):
        self.batch_size, self.rollout_max_epochs, self.ppg_max_epochs = batch_size, rollout_max_epochs, ppg_max_epochs
        self.gamma, self.gae_lambda, self.device = gamma, gae_lambda, device
        self.use_future, self.steps_into_future = use_future, steps_into_future
        
        self.oc_batch_size = batch_size
        self.memory_size, self.augmenter, self.parallel_processes = memory_size, augmenter, parallel_processes
        self.torch_rng = torch.Generator().manual_seed(seed)

        self.trajectories_length = []
        self._latest_observations, self._observations = [[] for i in range(num_parallel_envs)], []
        self.observation_qty, self.num_parallel_envs = 0, num_parallel_envs
        self.global_to_local_idx = {}
        self._initialize_trajectories()
        self._initialize_target_shapes(obs_shape = obs_shape, action_shape = action_shape)
        self._initialize_observations(random_dataset = random_dataset, dataset_preprocessor = dataset_preprocessor)
        
    def _initialize_trajectories(self):
        self._trajectories = {'obs': [[] for i in range(self.num_parallel_envs)], 
                              'reward': [[] for i in range(self.num_parallel_envs)], 
                              'done': [[] for i in range(self.num_parallel_envs)], 
                              'action': [[] for i in range(self.num_parallel_envs)],
                              'logprob': [[] for i in range(self.num_parallel_envs)],
                              'value': [[] for i in range(self.num_parallel_envs)]}

    def _initialize_target_shapes(self, obs_shape, action_shape):
        self.target_shapes = {'reward': (1,), 'done': (1,), 'logprob': (1,),
                              'value': (1,), 'advantage': (1,), 'return': (1,)}
        self.target_shapes['obs'] = obs_shape
        self.target_shapes['action'] = action_shape

    def _initialize_observations(self, random_dataset, dataset_preprocessor):
        i = 0
        obs_trajectory = []
        while i < self.memory_size:
            obs_trajectory.append(dataset_preprocessor(random_dataset['obss'][i]))
            if random_dataset['dones'][i]:
                self._observations.append(obs_trajectory)
                obs_trajectory = []
            i += 1
        self.observation_qty = i

    def _calculate_samples_per_phase(self, mode):
        if mode == 'rollout':
            return len(self.rollout['obs']) // self.batch_size * self.batch_size * self.rollout_max_epochs, self.batch_size
        elif mode == 'ppg':
            return self.observation_qty // self.oc_batch_size * self.oc_batch_size * self.ppg_max_epochs, self.oc_batch_size

    def save_transition(self, tran_dict):
        for key in tran_dict:
            for env_idx in range(self.num_parallel_envs):
                self._trajectories[key][env_idx].append(tran_dict[key][env_idx])
        
    def get_key(self, key):
        return self.rollout[key]
    
    def get_return_value(self):
        return self._trajectories['return'], self._trajectories['value']
    
    def _update_observations(self):
        for env_idx, traj in enumerate(self._trajectories['obs']):
            for idx_in_traj, obs in enumerate(traj):
                self._latest_observations[env_idx].append(obs.cpu())
                if self._trajectories['done'][env_idx][idx_in_traj]:
                    self.observation_qty += len(self._latest_observations[env_idx])
                    self._observations.append(self._latest_observations[env_idx])
                    self._latest_observations[env_idx] = []
        
        while self.observation_qty > self.memory_size:
            self.observation_qty -= len(self._observations[0])
            del self._observations[0]
    
    def _convert_trajectories_for_training(self):
        for key in self._trajectories.keys():
            if self._trajectories[key][0]:
                self._trajectories[key] = torch.stack([torch.stack(self._trajectories[key][i], axis = 0) \
                                          for i in range(self.num_parallel_envs)], axis = 0)
                self._trajectories[key] = torch.reshape(self._trajectories[key], 
                                                        self._trajectories[key].shape[:2] + self.target_shapes[key])
    
    def _precalculate_global_to_local_indexes(self):
        current_idx = 0
        for i in range(len(self._observations)):
            # Omit last step of each episode
            for j in range(len(self._observations[i])):
                self.global_to_local_idx[current_idx] = {'trajectory': i, 'step': j, 'traj_length': len(self._observations[i])}
                current_idx += 1
        
    def finalize_tensors_calculate_and_store_GAE(self, last_done, last_value):
        self._update_observations()
        self._convert_trajectories_for_training()
        self._precalculate_global_to_local_indexes()
        last_done = torch.unsqueeze(last_done, dim=-1)
        advantages = torch.zeros_like(self._trajectories['reward']).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self._trajectories['obs'].shape[1])):
            if t == self._trajectories['obs'].shape[1] - 1:
                nextnonterminal = 1.0 - last_done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - self._trajectories['done'][:, t + 1]
                nextvalues = self._trajectories['value'][:, t + 1]
            delta = self._trajectories['reward'][:, t] + self.gamma * nextvalues * nextnonterminal - self._trajectories['value'][:, t]
            advantages[:, t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + self._trajectories['value']
        self._trajectories['return'], self._trajectories['advantage'] = returns, advantages

    def convert_transitions_to_rollout(self):
        # TODO Check me, if i am sampling correct number of samples of correct batches
        self.rollout = {}
        for key in self._trajectories.keys():
            self.rollout[key] = deepcopy(self._trajectories[key].reshape((-1,) + self.target_shapes[key]))
        
        idxs = torch.cat([torch.randperm(len(self.rollout['obs']), generator = self.torch_rng) for i in range(self.rollout_max_epochs)])
        observation_pair_generator = self.get_obs_generator(mode='rollout')

        def batch_generator():
            for idx_subset in range(0, len(idxs), self.batch_size):
                batch = {}
                for key in self.rollout.keys():
                    batch[key] = self.rollout[key][idxs[idx_subset: idx_subset + self.batch_size]]
                start_obs, future_obs = next(observation_pair_generator)
                yield batch, start_obs, future_obs
        
        return batch_generator()
    
    def _prepare_idxs(self, num_samples):
        # Calculate number of samples for sampling while training with rollout and training with ppg
        # samples_per_rollout = self._calculate_iters_per_rollout() * self.batch_size
        # samples_per_ppg = self._calculate_iters_per_rollout() * self.oc_batch_size * self.ppg_mult
        sample_idxs = torch.multinomial(torch.ones(size = (self.observation_qty, ), dtype = torch.float32), 
                                        num_samples = num_samples, replacement = True, generator = self.torch_rng)
        
        starting_idxs, future_idxs = [], []
        for idx in sample_idxs:
            traj_idx, step_idx, traj_length = self.global_to_local_idx[idx.item()].values()
            step_delta = torch.multinomial(torch.ones(size = (min(self.steps_into_future, traj_length - step_idx), ),
                                                      dtype = torch.float32), 
                                                      num_samples = 1, generator = self.torch_rng).item()
            starting_idxs.append((traj_idx, step_idx)), future_idxs.append((traj_idx, step_idx + step_delta))
        return starting_idxs, future_idxs

    def get_obs_generator(self, mode):
        assert mode in ['rollout', 'ppg'], 'Only rollout and ppg modes are supported'
        num_samples, batch_size = self._calculate_samples_per_phase(mode)
        start_subdataset, finish_subdataset = [], []
        starting_idxs, future_idxs = self._prepare_idxs(num_samples)
        for idx in range(0, num_samples):
            start_traj_idx, start_obs_idx = starting_idxs[idx]
            finish_traj_idx, finish_obs_idx = future_idxs[idx]
            start_subdataset.append(self._observations[start_traj_idx][start_obs_idx])
            finish_subdataset.append(self._observations[finish_traj_idx][finish_obs_idx])
        if self.augmenter is not None:
            # Write code for parallel augmentation application for significant speedup
            raise NotImplementedError
        start_subdataset = torch.stack(start_subdataset, dim = 0)
        finish_subdataset = torch.stack(finish_subdataset, dim = 0)
        if self.use_future:
            for i in range(0, num_samples, batch_size):
                yield start_subdataset[i: i + batch_size].to(self.device), finish_subdataset[i: i + batch_size].to(self.device)
        else:
            for i in range(0, num_samples, batch_size):
                yield start_subdataset[i: i + batch_size].to(self.device), torch.full((batch_size,), torch.nan).to(self.device)
    
    def reset_trajectories(self):
        self.global_to_local_idx = {}
        self._initialize_trajectories()
        