import torch
import numpy as np
from copy import deepcopy

class OCRolloutBuffer():
    def __init__(self, gamma, gae_lambda, device, seed, num_parallel_envs, memory_size = 50_000):
        self.memory_size = memory_size
        self.gamma, self.gae_lambda, self.device = gamma, gae_lambda, device
        self.torch_rng = torch.Generator().manual_seed(seed)
        
        self.trajectories_length = []
        self.observations = [[[]] for i in range(num_parallel_envs)]
        self.observation_qty, self.trajs_deleted, self.num_parallel_envs = 0, 0, num_parallel_envs
        self.initialize_trajectories()
        
    def initialize_target_shapes(self, obs_shape, action_shape):
        self.target_shapes = {'reward': (1,), 'done': (1,), 'logprob': (1,),
                              'value': (1,), 'advantage': (1,), 'return': (1,)}
        self.target_shapes['obs'] = obs_shape
        self.target_shapes['action'] = action_shape

    def initialize_trajectories(self):
        self._trajectories = {'obs': [[] for i in range(self.num_parallel_envs)], 
                             'reward': [[] for i in range(self.num_parallel_envs)], 
                             'done': [[] for i in range(self.num_parallel_envs)], 
                             'action': [[] for i in range(self.num_parallel_envs)],
                             'logprob': [[] for i in range(self.num_parallel_envs)],
                             'value': [[] for i in range(self.num_parallel_envs)]}

    def save_transition(self, tran_dict):
        for key in tran_dict:
            for env_idx in range(self.num_parallel_envs):
                self._trajectories[key][env_idx].append(tran_dict[key][env_idx])

    def get_key(self, key):
        return self.rollout[key]
    
    def _update_observations(self):
        for env_idx, traj in enumerate(self._trajectories['obs']):
            self.observation_qty += len(traj)
            for idx_in_traj, obs in enumerate(traj):
                self.observations[env_idx][-1].append(obs)
                if self._trajectories['done'][env_idx][idx_in_traj]:
                    self.observations[env_idx].append([])
        
        while self.observation_qty > self.memory_size:
            self.observation_qty -= len(self.observations[self.trajs_deleted % self.num_parallel_envs][0])
            del self.observations[self.trajs_deleted % self.num_parallel_envs][0]
            self.trajs_deleted += 1
    
    def _convert_trajectories_for_training(self):
        for key in self._trajectories.keys():
            if self._trajectories[key][0]:
                self._trajectories[key] = torch.stack([torch.stack(self._trajectories[key][i], axis = 0) \
                                          for i in range(self.num_parallel_envs)], axis = 0)
                self._trajectories[key] = torch.reshape(self._trajectories[key], 
                                                        self._trajectories[key].shape[:2] + self.target_shapes[key])
        
    def finalize_tensors_calculate_and_store_GAE(self, last_done, last_value):
        self._update_observations()
        self._convert_trajectories_for_training()
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

    def convert_transitions_to_rollout(self, batch_size):
        self.rollout = {}
        for key in self._trajectories.keys():
            self.rollout[key] = deepcopy(self._trajectories[key].reshape((-1,) + self.target_shapes[key]))
            data_length = len(self.rollout[key])
        
        idxs = torch.randperm(data_length, generator = self.torch_rng)

        def batch_generator():
            for idx_subset in range(0, data_length, batch_size):
                batch = {}
                for key in self.rollout.keys():
                    batch[key] = self.rollout[key][idxs[idx_subset: idx_subset + batch_size]]
                yield batch
        
        return batch_generator()
    
    def get_return_value(self):
        return self._trajectories['return'], self._trajectories['value']
    
    def reset_trajectories(self):
        self.initialize_trajectories()
        