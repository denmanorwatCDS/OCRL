import numpy as np
import gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
import torch

class NormalizationWrapper(gym.Wrapper):
    def __init__(self, env, min_val, max_val):
        super().__init__(env)
        new_shape = tuple([env.observation_space.shape[2], *env.observation_space.shape[0:2]])
        box = gym.spaces.box.Box(shape = new_shape, high = np.ones(new_shape), 
                                 low = -1 * np.ones(new_shape), dtype = np.float32)
        self.observation_space = box
        self.min_val, self.max_val = min_val, max_val

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = (obs / 255.0 * (self.max_val - self.min_val)) + self.min_val
        obs = np.transpose(obs, (2, 0, 1))
        return obs, reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        obs = (obs / 255.0 * (self.max_val - self.min_val)) + self.min_val
        obs = np.transpose(obs, (2, 0, 1))
        return obs
