import datetime, functools
import comet_ml
import numpy as np
import torch
import math
import gym

from networks.distribution_networks import GaussianMLPTwoHeadedModule, GaussianMLPGlobalStdModule
from networks.mlp import MLPModule
from networks import pipeline
from utils.weight_initializer.xavier_init import xavier_normal
from RL.policies.policy import Policy
from ReplayBuffers.path_replay_buffer import PathBuffer
from RL.algos.ppo import PPO
from gym.vector import AsyncVectorEnv, SyncVectorEnv

from main import fetch_config, set_seed, fetch_activation, fetch_dist_type, prepare_batch

def build_policy_net(env, policy_net_config):
    obs_dim, action_dim = env.observation_space, np.array(env.action_space.shape[0])
    channels, obs_size = None, obs_dim

    module_obs_dim = obs_dim.shape[0]
    if policy_net_config.distribution.name == 'TwoHeaded':
        policy_module = GaussianMLPTwoHeadedModule(input_dim = module_obs_dim, output_dim = action_dim, 
                                               hidden_sizes = policy_net_config.hidden_sizes, 
                                               layer_normalization = policy_net_config.layer_normalization,
                                               hidden_nonlinearity = fetch_activation(policy_net_config.nonlinearity), 
                                               std_parameterization = policy_net_config.distribution.std_parameterization,
                                               max_std = np.exp(policy_net_config.distribution.max_logstd),
                                               init_std = np.exp(policy_net_config.distribution.starting_logstd),
                                               normal_distribution_cls = fetch_dist_type(policy_net_config.distribution.type),
                                               output_w_init = functools.partial(xavier_normal, gain=1.))
    elif policy_net_config.distribution.name == 'GlobalStd':
        policy_module = GaussianMLPGlobalStdModule(input_dim = module_obs_dim, output_dim = action_dim, 
                                               hidden_sizes = policy_net_config.hidden_sizes, 
                                               layer_normalization = policy_net_config.layer_normalization,
                                               hidden_nonlinearity = fetch_activation(policy_net_config.nonlinearity),
                                               std_parameterization = policy_net_config.distribution.std_parameterization,
                                               max_std = np.exp(policy_net_config.distribution.max_logstd),
                                               init_std = np.exp(policy_net_config.distribution.starting_logstd),
                                               normal_distribution_cls = fetch_dist_type(policy_net_config.distribution.type),
                                               output_w_init = functools.partial(xavier_normal, gain=1.))
    policy_module = pipeline.DictPipeline(policy_module)
    return Policy(name = policy_net_config.name, module = policy_module)

def build_v_net(env, v_net_config):
    obs_dim = env.observation_space
    channels, obs_size = None, obs_dim

    module_obs_dim = obs_dim.shape[0]

    v = MLPModule(input_dim = module_obs_dim, output_dim = 1,
                  hidden_sizes = v_net_config.hidden_sizes,
                  hidden_nonlinearity = fetch_activation(v_net_config.nonlinearity),
                  hidden_w_init = torch.nn.init.xavier_normal_,
                  hidden_b_init = torch.nn.init.zeros_,
                  output_nonlinearity = None,
                  output_w_init = torch.nn.init.xavier_normal_,
                  output_b_init = torch.nn.init.zeros_,
                  layer_normalization = v_net_config.layer_normalization)
    v = pipeline.DictPipeline(v)
    return v

class MuJoCoWrapper(gym.Wrapper):
    def __init__(self, env, timelimit = 1000):
        super().__init__(env)
        self.cur_step = 0
        self.max_steps = timelimit
        self.observation_space = env.observation_space
        self.observation_space.dtype = np.float32
        self.action_space = env.action_space
        self.action_space.dtype = np.float32

    def step(self, action):
        self.cur_step += 1
        obs, reward, done, info = self.env.step(action)
        if self.max_steps <= self.cur_step:
            done = True
        reward, done = np.array([reward], dtype=np.float32), np.array([done])
        obs = obs.astype(np.float32)
        return obs, reward, done, info

    def reset(self,):
        obs = self.env.reset()
        self.cur_step = 0
        return obs.astype(np.float32)

def make_env(seed):
    env = gym.make('HalfCheetah-v3')
    env.seed(seed=seed)
    env = MuJoCoWrapper(env)
    return env

def make_seeded_env(seed):
    return make_env(seed)

def run():
    config = fetch_config()
    g_start_time = int(datetime.datetime.now().timestamp())

    exp = comet_ml.start(project_name = 'PPO_test')
    exp.log_parameters(config)

    print('ARGS: ' + str(config))
    if config.globals.n_thread is not None:
        torch.set_num_threads(config.globals.n_thread)

    set_seed(config.globals.seed)
    env = make_env(42)

    policy = build_policy_net(env, policy_net_config = config.rl_algo.policy)

    optimizers = {
        'option_policy': torch.optim.Adam([
            {'params': policy.parameters(), 'lr': config.rl_algo.policy.lr},
        ])
    }
    
    rl_algo = None

    vf = build_v_net(env, v_net_config = config.rl_algo.value)
    optimizers.update({
        'vf': torch.optim.Adam([
        {'params': list(vf.parameters()), 'lr': config.rl_algo.value.lr}
    ])})
    rl_algo = PPO(vf, clip_coef = config.rl_algo.clip_coef, clip_vloss = config.rl_algo.clip_vloss, 
                    ent_coef = config.rl_algo.ent_coef, vf_coef = config.rl_algo.vf_coef, 
                    normalize_advantage = config.rl_algo.norm_adv, max_grad_norm = config.rl_algo.max_grad_norm, 
                    target_kl = config.rl_algo.target_kl, option_policy = policy, 
                    optimizers = {key: optimizers[key] for key in ['option_policy', 'vf']},
                    device = config.globals.device)
        
        
    rollout_buffer = PathBuffer(capacity_in_transitions = int(config.replay_buffer.policy.max_transitions), 
                                batch_size = config.replay_buffer.policy.batch_size, pixel_keys = [], 
                                discount = config.replay_buffer.discount, gae_lambda = config.replay_buffer.gae_lambda,
                                on_policy = rl_algo.on_policy)
        
    env.close()
    
    train_cycle(config.trainer_args, agent = rl_algo, rollout_buffer = rollout_buffer, 
                make_env_fn = make_seeded_env, seed = config.globals.seed, comet_logger = exp)
    
def train_cycle(trainer_config, agent, rollout_buffer, make_env_fn, seed, comet_logger):
    env = SyncVectorEnv([lambda: make_env_fn(seed = (seed + i)) for i in range(trainer_config.n_parallel)]) #, context='spawn')
    
    prev_cur_step, cur_step = 0, 0
    for i in range(trainer_config.n_epochs):
        agent.eval()
        trajs = collect_trajectories(env = env, agent = agent, trajectories_qty = trainer_config.traj_batch_size, 
                            trajectories_length = trainer_config.max_path_length)
        for traj in trajs:
            cur_step += len(traj['observations'])
        rollout_buffer.update_replay_buffer(trajs)

        agent.train()
        skill_optim_steps = trainer_config.skill_optimization_epochs if agent.on_policy \
            else trainer_config.trans_optimization_epochs
        
        for _ in range(skill_optim_steps):
            batch = rollout_buffer.sample_transitions()
            batch = prepare_batch(batch)
            logs = agent.optimize_op(batch)
        rollout_buffer.clear()
        
        if (prev_cur_step // trainer_config.log_frequency) < (cur_step // trainer_config.log_frequency):
            comet_logger.log_metrics(logs, step = cur_step)
        
        agent.eval()
        eval_env = make_seeded_env(1)
        if (prev_cur_step // trainer_config.eval_frequency) < (cur_step // trainer_config.eval_frequency):
            agent.eval()
            agent.option_policy._force_use_mode_actions = True
            is_terminated, returns = [], []
            for val_episode in range(10):
                video, dones, rewards = [], [], []
                prev_obs, done = eval_env.reset(), False
                while not (done):
                    agent_input = {'obs': np.expand_dims(prev_obs, axis=0)}
                    action, _ = agent.policy['option_policy'].get_actions(agent_input)
                    obs, reward, done, info = eval_env.step(action)
                    dones.append(done), rewards.append(reward)
                    prev_obs = obs
                dones = np.concatenate(dones, axis=0)
                returns.append(np.sum(rewards))
                is_terminated.append(np.max(dones))
            agent.option_policy._force_use_mode_actions = False
            comet_logger.log_metrics({'Mean done': np.mean(dones), 'Mean return': np.mean(returns)}, step=cur_step)
        
        prev_cur_step = cur_step
        eval_env.close()

def collect_trajectories(env, agent, trajectories_length, trajectories_qty = None, mode = 'train'):
    env_qty = len(env.env_fns)
    if isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
        pseudoepisodes = math.ceil(trajectories_qty / len(env.env_fns))
    else:
        pseudoepisodes = trajectories_qty
    generated_trajectories = [{} for i in range(trajectories_qty)]
    for pseudoepisode in range(pseudoepisodes):
        prev_obs = env.reset()
        prev_obs = np.transpose(prev_obs, [0, 3, 1, 2]) if len(prev_obs.shape) == 4 else prev_obs
        prev_dones = np.full((env_qty,), fill_value=False)
        for i in range(trajectories_length):
            agent_input = {'obs': prev_obs}
            action, action_info = agent.policy['option_policy'].get_actions(agent_input)
            next_obs, rewards, dones, env_infos = env.step(action)
            next_obs = np.transpose(next_obs, [0, 3, 1, 2]) if len(next_obs.shape) == 4 else next_obs
            if (i == trajectories_length - 1):
                dones = np.full((env_qty,), fill_value = True)
            for j, done in enumerate(prev_dones):
                if not done:
                    update_traj_with_array(generated_trajectories[j + pseudoepisode * env_qty], prev_obs[j: j+1], 
                                           'observations')
                    update_traj_with_array(generated_trajectories[j + pseudoepisode * env_qty], next_obs[j: j+1], 
                                           'next_observations')
                    update_traj_with_array(generated_trajectories[j + pseudoepisode * env_qty], action[j: j+1], 
                                           'actions')
                    update_traj_with_array(generated_trajectories[j + pseudoepisode * env_qty], rewards[j: j+1], 
                                           'rewards')
                    update_traj_with_array(generated_trajectories[j + pseudoepisode * env_qty], dones[j: j+1], 
                                           'dones')
                    update_traj_with_info(generated_trajectories[j + pseudoepisode * env_qty], env_infos[j], 'env_infos')
                    agent_info = {}
                    if mode == 'train':
                        agent_info.update({'log_probs': action_info['log_prob'][j], 
                                           'pre_tanh_actions': action_info['pre_tanh_value'][j],
                                           'options': np.zeros_like(action_info['log_prob'][j].shape, dtype = np.float32)})
                    update_traj_with_info(generated_trajectories[j + pseudoepisode * env_qty], agent_info, 'agent_infos')
            prev_obs = next_obs
            prev_dones = np.logical_or(prev_dones, dones)
    
    generated_trajectories = generated_trajectories[:trajectories_qty]
    if agent.on_policy and mode == 'train':
        prev_obs = torch.tensor([generated_trajectories[i]['observations'] for i in range(len(generated_trajectories))])
        next_obs = torch.tensor([generated_trajectories[i]['next_observations'] for i in range(len(generated_trajectories))])
        prev_obs = {'obs': prev_obs.reshape((-1,) + tuple(prev_obs.shape[2:])).to(agent.device)}
        next_obs = {'obs': next_obs.reshape((-1,) + tuple(next_obs.shape[2:])).to(agent.device)}
        values = agent.critic(prev_obs).detach().cpu().numpy().astype(np.float32).reshape((trajectories_qty, -1))
        next_values = agent.critic(next_obs).detach().cpu().numpy().astype(np.float32).reshape((trajectories_qty, -1))
        agent_info.update({'values': agent.critic(prev_obs).detach().cpu().numpy(),
                           'next_values': agent.critic(next_obs).detach().cpu().numpy()})
        for i in range(len(generated_trajectories)):
            generated_trajectories[i]['agent_infos']['values'] = values[i]
            generated_trajectories[i]['agent_infos']['next_values'] = next_values[i]
    return generated_trajectories

def update_traj_with_info(target_dict, infos, info_dict_name):
    if info_dict_name not in target_dict:
        target_dict[info_dict_name] = {}
        for key in infos.keys():
            data = None
            if isinstance(infos[key], np.ndarray):
                if len(infos[key].shape) > 0:
                    data = np.expand_dims(infos[key].copy(), axis = 0)
                else:
                    data = infos[key].copy()
            elif isinstance(infos[key], (np.int64, np.float32, float, bool)):
                data = np.array([infos[key]])
            else:
                assert False, 'Something went horribly wrong...'
            target_dict[info_dict_name][key] = data
    else:
        for key in infos.keys():
            if isinstance(infos[key], (np.float32, float, bool)):
                target_dict[info_dict_name][key] = np.append(target_dict[info_dict_name][key], infos[key])
            elif len(infos[key].shape) > 0:
                target_dict[info_dict_name][key] = np.append(target_dict[info_dict_name][key], 
                                                         np.expand_dims(infos[key], axis = 0), axis = 0)
            else:
                target_dict[info_dict_name][key] = np.append(target_dict[info_dict_name][key], infos[key])

def update_traj_with_array(target_dict, array, array_key):
    if array_key not in target_dict.keys():
        target_dict[array_key] = array
    else:
        target_dict[array_key] = np.append(target_dict[array_key], array, axis=0)


if __name__ == '__main__':
    run()