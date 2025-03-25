"""An environment wrapper that normalizes action, observation and reward."""
import akro
import gym
import gym.spaces
import gym.spaces.utils
import numpy as np

from envs.utils.akro_wrapper import AkroWrapperTrait

class ConsistentNormalizedEnv(AkroWrapperTrait, gym.Wrapper):
    def __init__(
            self,
            env,
            expected_action_scale=1.,
            flatten_obs=True,
            normalize_obs=True,
            mean=None,
            std=None,
    ):
        super().__init__(env)

        self._normalize_obs = normalize_obs
        self._expected_action_scale = expected_action_scale
        self._flatten_obs = flatten_obs

        self._obs_mean = np.full(env.observation_space.shape, 0 if mean is None else mean)
        self._obs_var = np.full(env.observation_space.shape, 1 if std is None else std ** 2)

        self._cur_obs = None

        if isinstance(self.env.action_space, gym.spaces.Box):
            self.action_space = akro.Box(low=-self._expected_action_scale,
                                         high=self._expected_action_scale,
                                         shape=self.env.action_space.shape)
        else:
            self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def _apply_normalize_obs(self, obs):
        normalized_obs = (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)
        return normalized_obs

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._cur_obs = obs

        if self._normalize_obs:
            obs = self._apply_normalize_obs(obs)

        if self._flatten_obs:
            obs = gym.spaces.utils.flatten(self.env.observation_space, obs)

        return obs

    def step(self, action, **kwargs):
        if isinstance(self.env.action_space, gym.spaces.Box):
            # rescale the action when the bounds are not inf
            lb, ub = self.env.action_space.low, self.env.action_space.high
            if np.all(lb != -np.inf) and np.all(ub != -np.inf):
                scaled_action = lb + (action + self._expected_action_scale) * (
                        0.5 * (ub - lb) / self._expected_action_scale)
                scaled_action = np.clip(scaled_action, lb, ub)
            else:
                scaled_action = action
        else:
            scaled_action = action

        next_obs, reward, done, info = self.env.step(scaled_action, **kwargs)
        info['original_observations'] = self._cur_obs
        info['original_next_observations'] = next_obs

        self._cur_obs = next_obs

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)

        if self._flatten_obs:
            next_obs = gym.spaces.utils.flatten(self.env.observation_space, next_obs)

        return next_obs, reward, done, info

consistent_normalize = ConsistentNormalizedEnv

def get_normalizer_preset(normalizer_type):
    # Precomputed mean and std of the state dimensions from 10000 length-50 random rollouts (without early termination)
    if normalizer_type == 'off':
        normalizer_mean = np.array([0.])
        normalizer_std = np.array([1.])
    elif normalizer_type == 'half_cheetah_preset':
        normalizer_mean = np.array(
            [-0.07861924, -0.08627162, 0.08968642, 0.00960849, 0.02950368, -0.00948337, 0.01661406, -0.05476654,
             -0.04932635, -0.08061652, -0.05205841, 0.04500197, 0.02638421, -0.04570961, 0.03183838, 0.01736591,
             0.0091929, -0.0115027])
        normalizer_std = np.array(
            [0.4039283, 0.07610687, 0.23817, 0.2515473, 0.2698137, 0.26374814, 0.32229397, 0.2896734, 0.2774097,
             0.73060024, 0.77360505, 1.5871304, 5.5405455, 6.7097645, 6.8253727, 6.3142195, 6.417641, 5.9759197])
    elif normalizer_type == 'ant_preset':
        normalizer_mean = np.array(
            [0.00486117, 0.011312, 0.7022248, 0.8454677, -0.00102548, -0.00300276, 0.00311523, -0.00139029,
             0.8607109, -0.00185301, -0.8556998, 0.00343217, -0.8585605, -0.00109082, 0.8558013, 0.00278213,
             0.00618173, -0.02584622, -0.00599026, -0.00379596, 0.00526138, -0.0059213, 0.27686235, 0.00512205,
             -0.27617684, -0.0033233, -0.2766923, 0.00268359, 0.27756855])
        normalizer_std = np.array(
            [0.62473416, 0.61958003, 0.1717569, 0.28629342, 0.20020866, 0.20572574, 0.34922406, 0.40098143,
             0.3114514, 0.4024826, 0.31057045, 0.40343934, 0.3110796, 0.40245822, 0.31100526, 0.81786263, 0.8166509,
             0.9870919, 1.7525449, 1.7468817, 1.8596431, 4.502961, 4.4070187, 4.522444, 4.3518476, 4.5105968,
             4.3704205, 4.5175962, 4.3704395])
    else:
        raise NotImplementedError

    return normalizer_mean, normalizer_std