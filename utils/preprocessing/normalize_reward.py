import copy
import numpy as np
from math import sqrt

def clip(value, max_value):
    if value < max_value:
        return value
    return max_value

class NormalizeReward:
    """Tracks the mean, variance and count of values."""

    def __init__(self, gamma, buffer_size = 50_000):
        """Tracks the mean, variance and count of values."""
        self.values = np.zeros((buffer_size), "float32")
        self.buffer_size = buffer_size
        self.total_samples_processed = 0
        self.gamma = gamma
        self.mean, self.var = 0, 1

    def modify_reward(self, rewards):
        assert len(rewards.shape) == 2, '3 dimensions expected: [batch, time]'
        rets = copy.deepcopy(rewards)
        for t in range(1, rewards.shape[1]):
            rets[:, t] = rets[:, t - 1] * self.gamma + rewards[:, t]
        self._update_running_mean_var(rets)
        rewards = rewards / sqrt(self.var + 1e-04)
        return rewards

    def _update_running_mean_var(self, x):
        """Updates the mean, var and count from a batch of samples."""
        assert len(x.shape) == 2, '3 dimensions expected: [batch, time]'
        samples_qty = x.shape[0] * x.shape[1]
        substitution_idxs = np.arange(self.total_samples_processed, self.total_samples_processed + samples_qty) \
            % self.buffer_size
        self.values[substitution_idxs] = x.reshape(-1)
        self.total_samples_processed += samples_qty
        self.mean = np.mean(self.values[:clip(self.total_samples_processed, self.buffer_size)], axis=0)
        self.var = np.var(self.values[:clip(self.total_samples_processed, self.buffer_size)], axis=0)

    def get_mean(self):
        return copy.deepcopy(self.mean)

    def get_var(self):
        return copy.deepcopy(self.var)