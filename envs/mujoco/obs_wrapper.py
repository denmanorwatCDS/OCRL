from akro import Box
from envs.utils.akro_wrapper import AkroWrapperTrait
from gym import Wrapper
import numpy as np

class ExpanderWrapper(AkroWrapperTrait, Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = self.env.action_space
        akro_box = Box(low = np.expand_dims(self.env.observation_space.low, axis = 0),
                       high = np.expand_dims(self.env.observation_space.low, axis = 0),
                       dtype = np.float32)
        self.observation_space = akro_box
    
    def reset(self):
        obs = self.env.reset()
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis = 0).astype(np.float32)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis = 0).astype(np.float32)
        return obs, reward, done, info
    
    def render_step(self, action):
        next_obs, reward, done, env_info, img = self.env.render_step(action)
        return np.expand_dims(next_obs, axis = 0), reward, done, env_info, img
    