import gym
import cv2
import numpy as np
from gym.wrappers import AtariPreprocessing

class PatchedAtariPreprocessing(AtariPreprocessing):
    def reset(self, **kwargs):
        # NoopReset
        self.env.reset(**kwargs)
        noops = (
            self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
            if self.noop_max > 0
            else 0
        )
        for _ in range(noops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset(**kwargs)

        self.lives = self.ale.lives()
        if self.grayscale_obs:
            self.ale.getScreenGrayscale(self.obs_buffer[0])
        else:
            self.ale.getScreenRGB(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)
        return self._get_obs()

class Pong(gym.Env):
    metadata = {"render.modes": ["rgb_array", "state", "image", "mask"]}
    def __init__(self, config, seed):
        # See https://gymnasium.farama.org/v0.26.3/environments/atari/
        # Due to the usage of AtariPreprocessing, disable frameskip in original env
        self.env = gym.make('ALE/Pong-v5', obs_type='rgb', full_action_space='false', 
                            render_mode='rgb_array')
        self.env = PatchedAtariPreprocessing(self.env, screen_size=config['obs_size'], 
                                             grayscale_obs=False, frame_skip=1)
        self.env.reset(seed=seed)
        self.obs_size = config['obs_size']

    def step(self, action):
        return self.env.step(action)

    def render_mask(self):
        orig_env = self.env.unwrapped
        obs = orig_env.render(mode='rgb_array')
        # Remove scores and white stripes from mask calculation
        # Background color: [144, 72, 17]
        obs[:34, :] = np.array([144, 72, 17])
        obs[194:210, :] = np.array([144, 72, 17])
        obs_resized = cv2.resize(obs, (self.obs_size, self.obs_size), interpolation=cv2.INTER_NEAREST_EXACT)
        classes = np.unique(np.reshape(obs_resized, (-1, 3)), axis=0)
        # Move background class to last
        rearanged_classes = []
        for class_type in classes:
            if np.all(class_type != np.array([144, 72, 17])):
                rearanged_classes.append(class_type)
        rearanged_classes.append([144, 72, 17])
        classes = np.array(rearanged_classes)
        masks = np.zeros(obs_resized.shape[0:2])
        for i, type in enumerate(classes):
            masks = np.where(np.all(obs_resized == type, axis = -1), np.zeros(obs_resized.shape[0:2]) + i, masks)
        return masks

    def reset(self, seed=None):
        return self.env.reset(seed=seed)
    
    def render(self, mode):
        assert mode=='rgb_array', 'It should work for rgb_array mode, unknown for others'
        return self.env._get_obs()

    @property
    def action_space(self):
        """
        Return Gym's action space.
        """
        return self.env.action_space

    @property
    def observation_space(self):
        """
        Return Gym's observation space.
        """
        return self.env.observation_space