import gym
import numpy as np

class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation(
    gym.ObservationWrapper,
):
    """Normalizes observations to be centered at the mean with unit variance.

    The property :attr:`update_running_mean` allows to freeze/continue the running mean calculation of the observation
    statistics. If ``True`` (default), the ``RunningMeanStd`` will get updated every time ``step`` or ``reset`` is called.
    If ``False``, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.NormalizeObservation`.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.

    Example:
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> obs, info = env.reset(seed=123)
        >>> term, trunc = False, False
        >>> while not (term or trunc):
        ...     obs, _, term, trunc, _ = env.step(1)
        ...
        >>> obs
        array([ 0.1511158 ,  1.7183299 , -0.25533703, -2.8914354 ], dtype=float32)
        >>> env = gym.make("CartPole-v1")
        >>> env = NormalizeObservation(env)
        >>> obs, info = env.reset(seed=123)
        >>> term, trunc = False, False
        >>> while not (term or trunc):
        ...     obs, _, term, trunc, _ = env.step(1)
        >>> obs
        array([ 2.0059888,  1.5676788, -1.9944268, -1.6120394], dtype=float32)

    Change logs:
     * v0.21.0 - Initially add
     * v1.0.0 - Add `update_running_mean` attribute to allow disabling of updating the running mean / standard, particularly useful for evaluation time.
        Casts all observations to `np.float32` and sets the observation space with low/high of `-np.inf` and `np.inf` and dtype as `np.float32`
    """

    def __init__(self, env, epsilon: float = 1e-8):
        """This wrapper will normalize observations such that each observation is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.ObservationWrapper.__init__(self, env)

        assert env.observation_space.shape is not None
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

        self.obs_rms = RunningMeanStd(
            shape=self.observation_space.shape
        )
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the observation statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the observation statistics."""
        self._update_running_mean = setting

    def observation(self, observation):
        """Normalises the observation using the running mean and variance of the observations."""
        if self._update_running_mean:
            self.obs_rms.update(np.array([observation]))
        return np.float32(
            (observation - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        )
    
class NormalizeReward(gym.core.Wrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, dones, infos = self.env.step(action)
        infos['old_reward'] = rews
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma * (1 - dones) + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, dones, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)