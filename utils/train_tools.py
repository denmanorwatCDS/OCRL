import envs
import gym
import random
import torch
import numpy as np

class NormalizationWrapper(gym.Wrapper):
    def __init__(self, env, obs_preprocessor):
        super().__init__(env)
        new_shape = tuple([env.observation_space.shape[2], *env.observation_space.shape[0:2]])
        box = gym.spaces.box.Box(shape = new_shape, high = np.ones(new_shape), 
                                 low = -1 * np.ones(new_shape), dtype = np.float32)
        self.observation_space = box
        self.obs_preprocessor = obs_preprocessor

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.obs_preprocessor(obs)
        obs = np.transpose(obs, (2, 0, 1))
        return obs, reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        obs = self.obs_preprocessor(obs)
        obs = np.transpose(obs, (2, 0, 1))
        return obs
    
class ActionSqueezeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        return self.env.step(np.squeeze(action))

def infer_obs_action_shape(envs):
    # Due to vectorization, env will be represented by spaces.tuple.Tuple
    # Don't work with single_observation_space, works with observation_space
    obs_shape = envs.single_observation_space.shape
    if isinstance(envs.single_action_space, gym.spaces.discrete.Discrete):
        is_discrete = True
        agent_action_data = envs.single_action_space.n
        action_shape = (1,)    
    elif isinstance(envs.single_action_space, gym.spaces.Box):
        is_discrete = False
        act = envs.single_action_space.sample()
        assert len(act.shape) == 1, 'It is expected that action is a continuous vector'
        agent_action_data, action_shape = act.shape[0], act.shape

    if action_shape is None:
        assert False, 'WhatDaFaq?'

    return obs_shape, is_discrete, agent_action_data, action_shape

def make_env(env_config, gamma, ocr_min_val, ocr_max_val, seed = 0, rank = 0):
    if env_config.env == 'HalfCheetah-v3':
        env = gym.make("HalfCheetah-v3")
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma = gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    elif env_config.env == 'CartPole-v1':
        env = gym.make('CartPole-v1')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ActionSqueezeWrapper(env)
        env = gym.wrappers.NormalizeReward(env, gamma = gamma)
    else:
        env = getattr(envs, env_config.env)(env_config, seed + rank)
        env = NormalizationWrapper(env, min_val = ocr_min_val, max_val = ocr_max_val)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    return env

def get_model_name(dataset_name, image_size, model_name, save_name):
    dir_name = '/'.join(['models', image_size, dataset_name])
    model_name = '/'.join([dir_name, '_'.join([model_name, save_name])])
    return dir_name, model_name

def update_curves_(curve_dict, metrics):
    if not curve_dict:
        for metric_name, metric_value in metrics.items():
            curve_dict[metric_name] = [metric_value]
    else:
        for metric_name, metric_value in metrics.items():
            curve_dict[metric_name].append(metric_value)

def get_uint_to_float(min_val, max_val):
    torch_uint_to_float = lambda x: (x.to(torch.float32) / 255 * (max_val - min_val) + min_val)
    numpy_uint_to_float = lambda x: (x.astype(np.float32) / 255 * (max_val - min_val) + min_val)
    return torch_uint_to_float, numpy_uint_to_float

def get_float_to_uint(min_val, max_val):
    return lambda x: ((x - min_val) / (max_val - min_val) * 255).to(torch.uint8)

def stop_oc_optimizer_(oc_model, optimizer_config):
    for module_name in oc_model.get_grouped_parameters().keys():
        optim_name = '_'.join([module_name, 'optimizer'])
        del optimizer_config[optim_name]