import argparse, datetime, functools, sys
import comet_ml
import numpy as np
import omegaconf
import pathlib
import torch
import math

from envs.utils.consistent_normalized_env import consistent_normalize, get_normalizer_preset
from envs.mujoco.obs_wrapper import ExpanderWrapper
from ReplayBuffers.path_replay_buffer import PathBuffer
from RL.skill_model.metra_v2 import METRA
from RL.policies.sac import SAC
from networks.utils.parameter import ParameterModule
from gym.vector import AsyncVectorEnv, SyncVectorEnv
from eval_utils.eval_utils import get_option_colors, draw_2d_gaussians, StatisticsCalculator
import matplotlib.pyplot as plt
import matplotlib

def set_seed(seed):
    import random
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fetch_config():
    parser = argparse.ArgumentParser(prog='Metra')
    parser.add_argument('--default_config')
    parser.add_argument('--pipeline_config')
    parser.add_argument('--env_config')
    args = parser.parse_args()

    config_folder = str(pathlib.Path(__file__).parent.resolve()) + '/configs'
    rl_config_path = config_folder + '/rl_algos/' + args.default_config
    rl_config = omegaconf.OmegaConf.load(rl_config_path)
    algo_name = rl_config.rl_algo.name.lower()

    pipeline_config_path = config_folder + '/pipelines/' + args.pipeline_config
    pipeline_config = omegaconf.OmegaConf.load(pipeline_config_path)
    rl_config.merge_with(pipeline_config)

    env_config_path = config_folder + '/' + args.env_config
    env_config = omegaconf.OmegaConf.load(env_config_path)
    rl_config.merge_with(env_config)

    return rl_config

def make_env(env_name, env_kwargs, max_path_length, seed, frame_stack, normalizer_type):
    if env_name == 'maze':
        from envs.maze_env import MazeEnv
        env = MazeEnv(
            max_path_length=max_path_length,
            action_range=0.2,
        )
        env = ExpanderWrapper(env)
    elif env_name == 'half_cheetah':
        from envs.mujoco.half_cheetah_env import HalfCheetahEnv
        env = HalfCheetahEnv(render_hw = 100)
        env.seed(seed = seed)
    elif env_name == 'ant':
        from envs.mujoco.ant_env import AntEnv
        env = AntEnv(render_hw = 100)
        env = ExpanderWrapper(env)
        env.seed(seed = seed)
    elif env_name == 'gripper':
        from envs.mujoco.gripper_env import MultipleFetchPickAndPlaceEnv
        env = MultipleFetchPickAndPlaceEnv(seed = seed, obs_type = 'state', object_qty = env_kwargs.object_qty,
                                           object_names = env_kwargs.object_names)
        env = ExpanderWrapper(env)
    elif env_name == 'pixel_gripper':
        from envs.mujoco.gripper_env import MultipleFetchPickAndPlaceEnv
        env = MultipleFetchPickAndPlaceEnv(seed = seed, obs_type = 'pixels', object_qty = env_kwargs.object_qty,
                                           object_names = env_kwargs.object_names)
    elif env_name == 'decoupled_gripper':
        from envs.mujoco.gripper_env import MultipleFetchPickAndPlaceEnv
        env = MultipleFetchPickAndPlaceEnv(seed = seed, obs_type = 'decoupled_state', object_qty = env_kwargs.object_qty,
                                           object_names = env_kwargs.object_names)
    elif env_name.startswith('dmc'):
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
        if env_name == 'dmc_cheetah':
            env = dmc.make('cheetah_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=seed)
            env = RenderWrapper(env)
        elif env_name == 'dmc_quadruped':
            env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=seed)
            env = RenderWrapper(env)
        elif env_name == 'dmc_humanoid':
            env = dmc.make('humanoid_run_color', obs_type='states', frame_stack=1, action_repeat=2, seed=seed)
            env = RenderWrapper(env)
        else:
            raise NotImplementedError
    elif env_name == 'kitchen':
        sys.path.append('lexa')
        from envs.lexa.mykitchen import MyKitchenEnv
        assert seed is None, 'For some strange reason, this environment does not have any seed...'
        env = MyKitchenEnv(log_per_goal=True)
    else:
        raise NotImplementedError

    if frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
        env = FrameStackWrapper(env, frame_stack)

    normalizer_kwargs = {}

    if normalizer_type == 'off':
        env = consistent_normalize(env, flatten_obs = False, normalize_obs = False, **normalizer_kwargs)
    elif normalizer_type == 'squashed':
        env = consistent_normalize(env, flatten_obs = False, normalize_obs = True, 
                                   mean = 255. / 2, std = 255. / 2)
    elif normalizer_type == 'preset':
        normalizer_name = env_name
        normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        env = consistent_normalize(env, flatten_obs = False, normalize_obs = True, mean = normalizer_mean, std = normalizer_std, 
                                   **normalizer_kwargs)

    return env
    
def run():
    config = fetch_config()
    g_start_time = int(datetime.datetime.now().timestamp())

    exp = comet_ml.start(project_name = 'metra')
    exp.log_parameters(config)

    print('ARGS: ' + str(config))
    if config.globals.n_thread is not None:
        torch.set_num_threads(config.globals.n_thread)

    set_seed(config.globals.seed)
    # TODO check that seeding is correct, meaning that observations must be uncorrelated!
    make_seeded_env = functools.partial(make_env, env_name = config.env.name, 
                                        env_kwargs = config.env.env_kwargs,
                                        max_path_length = config.env.max_path_length,
                                        frame_stack = config.env.frame_stack, 
                                        normalizer_type = config.env.normalizer_type)
    env = make_seeded_env(seed = config.globals.seed)
    
    # TODO change shape
    rl_algo = SAC(name = 'SAC', obs_length = env.observation_space.shape[1], task_length = config.skill.dim_option, 
                  action_length = env.action_space.shape[0], actor_config = config.rl_algo.policy, 
                  critic_config = config.rl_algo.critics, pooler_config = config.rl_algo.slot_pooler,
                  alpha = config.rl_algo.alpha.value, tau = config.rl_algo.tau, scale_reward = config.rl_algo.scale_reward,
                  env_spec = env, target_coef = config.rl_algo.target_coef, device = config.globals.device,
                  discount = config.rl_algo.discount,
                  actor_lr = config.rl_algo.policy.lr, critic_lr = config.rl_algo.critics.lr, 
                  pooler_lr = config.rl_algo.slot_pooler.lr, log_alpha_lr = config.rl_algo.alpha.lr,
                  actor_wd = config.rl_algo.policy.wd, critic_wd = config.rl_algo.critics.wd, 
                  pooler_wd = config.rl_algo.slot_pooler.wd, log_alpha_wd = config.rl_algo.alpha.wd)
    
    replay_buffer = PathBuffer(capacity_in_transitions = int(config.replay_buffer.common.max_transitions), 
                               batch_size = config.replay_buffer.common.batch_size, pixel_keys = {}, 
                               discount = config.replay_buffer.common.discount, gae_lambda = None,
                               on_policy = rl_algo.on_policy)

    metra = METRA(obs_length = env.observation_space.shape[1], pooler_config = config.skill.slot_pooler, 
                  traj_encoder_config = config.skill.trajectory_encoder, dist_predictor_config = None,
                  pooler_lr = config.skill.slot_pooler.lr, traj_lr = config.skill.trajectory_encoder.lr, 
                  dist_lr = None, dual_lam_lr = config.skill.dual_lr, 
                  pooler_wd = config.skill.slot_pooler.wd, traj_wd = config.skill.trajectory_encoder.wd,
                  dist_wd = None, dual_lam_wd = config.skill.dual_wd, 
                  dist_predictor_name = config.skill.dual_dist_name, dual_lam = config.skill.dual_lam,
                  option_size = config.skill.dim_option, discrete = config.skill.discrete, 
                  unit_length = config.skill.unit_length, device = config.globals.device, dual_reg = config.skill.dual_reg,
                  dual_slack = config.skill.dual_slack)
    env.close()
    
    train_cycle(config.trainer_args, agent = rl_algo, skill_model = metra, replay_buffer = replay_buffer,
                rollout_buffer = None, running_std = None, make_env_fn = make_seeded_env, 
                seed = config.globals.seed, comet_logger = exp)

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
            elif isinstance(infos[key], (np.int64, np.float32, float)):
                data = np.array([infos[key]])
            else:
                assert False, 'Something went horribly wrong...'
            target_dict[info_dict_name][key] = data
    else:
        for key in infos.keys():
            if isinstance(infos[key], (np.float32, float)):
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


def collect_trajectories(env, agent, trajectories_length, skills_per_traj = None, n_objects = None, trajectories_qty = None, 
                         skill_model = None, options_and_obj_idxs = None, mode = 'train'):
    if options_and_obj_idxs is None:
        options_and_obj_idxs = skill_model._get_train_trajectories_kwargs(batch_size = trajectories_qty, 
                                                                          traj_len = trajectories_length,
                                                                          skills_per_traj = skills_per_traj, 
                                                                          n_objects = n_objects)
    else:
        assert (trajectories_qty is None) and (skill_model is None) and (n_objects is None) and (skills_per_traj is None),\
        """ It is expected, that when using options, there are one trajectory per each option, 
        and there is no need to sample new options """
        trajectories_qty = len(options_and_obj_idxs)
    options_and_obj_idxs = np.array(options_and_obj_idxs)
    
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
            opt_and_obj = options_and_obj_idxs[pseudoepisode * env_qty: (pseudoepisode + 1) * env_qty, i]
            cur_options = np.stack([opt_and_obj[i]['options'] for i in range(env_qty)], axis=0)
            cur_obj_idx = np.stack([opt_and_obj[i]['obj_idxs'] for i in range(env_qty)], axis=0)
            obs_and_option = {'obs': prev_obs, 'options': cur_options, 'obj_idxs': cur_obj_idx}
            action, action_info = agent.get_actions(prev_obs, cur_obj_idx, cur_options)
            next_obs, rewards, dones, env_infos = env.step(action)
            next_obs = np.transpose(next_obs, [0, 3, 1, 2]) if len(next_obs.shape) == 4 else next_obs
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
                    agent_info.update({'options': cur_options[j], 'obj_idxs': cur_obj_idx[j]})
                    if mode == 'train':
                        agent_info.update({'log_probs': action_info['log_prob'][j], 
                                           'pre_tanh_actions': action_info['pre_tanh_value'][j]})
                    update_traj_with_info(generated_trajectories[j + pseudoepisode * env_qty], agent_info, 'agent_infos')
            prev_obs = next_obs
            prev_dones = np.logical_or(prev_dones, dones)
    
    generated_trajectories = generated_trajectories[:trajectories_qty]
    if agent.on_policy and mode == 'train':
        prev_obs = torch.tensor(np.array([generated_trajectories[i]['observations'] for i in range(len(generated_trajectories))]))
        next_obs = torch.tensor(np.array([generated_trajectories[i]['next_observations'] for i in range(len(generated_trajectories))]))
        options = torch.tensor(np.array([generated_trajectories[i]['agent_infos']['options'] for i in range(len(generated_trajectories))]))
        obj_idxs = torch.tensor(np.array([generated_trajectories[i]['agent_infos']['obj_idxs'] for i in range(len(generated_trajectories))]))
        prev_obs = {'obs': prev_obs.reshape((-1,) + tuple(prev_obs.shape[2:])).to(agent.device), 
                    'options': options.reshape((-1,) + tuple(options.shape[2:])).to(agent.device), 
                    'obj_idxs': obj_idxs.reshape((-1,) + tuple(obj_idxs.shape[2:])).to(agent.device)}
        next_obs = {'obs': next_obs.reshape((-1,) + tuple(next_obs.shape[2:])).to(agent.device), 
                    'options': options.reshape((-1,) + tuple(options.shape[2:])).to(agent.device), 
                    'obj_idxs': obj_idxs.reshape((-1,) + tuple(obj_idxs.shape[2:])).to(agent.device)}
        values = agent.critic(prev_obs).detach().cpu().numpy().astype(np.float32).reshape((trajectories_qty, -1))
        next_values = agent.critic(next_obs).detach().cpu().numpy().astype(np.float32).reshape((trajectories_qty, -1))
        for i in range(len(generated_trajectories)):
            generated_trajectories[i]['agent_infos']['values'] = values[i]
            generated_trajectories[i]['agent_infos']['next_values'] = next_values[i]
    return generated_trajectories

def render_trajectories(env, agent, options, trajectories_length):
    videos = []
    trajectories_qty = len(options)

    render_step = True
    if env.is_image:
        render_step = False
    
    for pseudoepisode in range(trajectories_qty):
        videos.append([])
        cur_option = np.array([options[pseudoepisode]['options']]).astype(np.float32)
        cur_obj = np.array([options[pseudoepisode]['obj_idxs']])
        prev_obs = env.reset()
        prev_done = False

        for i in range(trajectories_length):
            prev_obs = np.expand_dims(prev_obs, 0).astype(np.float32)
            prev_obs = np.transpose(prev_obs, [0, 3, 1, 2]) if len(prev_obs.shape) == 4 else prev_obs
            obs_and_option_and_obj = {'obs': prev_obs, 'options': cur_option, 'obj_idxs': cur_obj}
            action, action_info = agent.policy['option_policy'].get_actions(obs_and_option_and_obj)
            action = action[0]
            if render_step:
                next_obs, reward, done, env_info, img = env.render_step(action)
            else:
                next_obs, reward, done, env_info = env.step(action)
                img = (next_obs.copy() * 255/2 + 255/2).astype(np.uint8)
            videos[-1].append(img)

            prev_obs = next_obs
    return np.array(videos)

def prepare_batch(batch, device = 'cuda'):
    data = {}
    for key, value in batch.items():
        data[key] = torch.from_numpy(value).to(device)
    return data

def train_cycle(trainer_config, agent, skill_model, replay_buffer, rollout_buffer, running_std, make_env_fn, seed, 
                comet_logger):
    n_objects = make_env_fn(seed = 0).n_obj
    env = AsyncVectorEnv([lambda: make_env_fn(seed = (seed + i)) for i in range(trainer_config.n_parallel)], context='spawn')
    eval_env = AsyncVectorEnv([lambda: make_env_fn(seed = (seed + i)) for i in range(trainer_config.n_parallel)], context='spawn')
    
    prev_cur_step, cur_step = 0, 0
    for i in range(trainer_config.n_epochs):
        agent.eval()
        trajs = collect_trajectories(env = env, agent = agent, skill_model = skill_model, 
                                     skills_per_traj = trainer_config.skills_per_trajectory, n_objects = n_objects,
                                     trajectories_qty = trainer_config.traj_batch_size, 
                                     trajectories_length = trainer_config.max_path_length)
        prev_cur_step = cur_step
        for traj in trajs:
            cur_step += len(traj['observations'])
        replay_buffer.update_replay_buffer(trajs)
        if (replay_buffer.n_transitions_stored < trainer_config.transitions_before_training):
            replay_buffer.delete_recent_paths()
            continue

        agent.train()
        skill_optim_steps = trainer_config.skill_optimization_epochs if agent.on_policy \
            else trainer_config.trans_optimization_epochs
        
        skill_stats = StatisticsCalculator('skill')
        policy_stats = StatisticsCalculator('policy')

        for i in range(skill_optim_steps):
            batch = replay_buffer.sample_transitions()
            batch = prepare_batch(batch)
            logs, rewards = skill_model.train_components(observations = batch['obs'], 
                                                         next_observations = batch['next_obs'],
                                                         options = batch['options'],
                                                         obj_idxs = batch['obj_idxs'])
            skill_stats.save_iter(logs)
            if not agent.on_policy:
                logs = agent.optimize_op(observations = batch['obs'], next_observations = batch['next_obs'], 
                                         obj_idxs = batch['obj_idxs'], options = batch['options'], 
                                         next_options = batch['next_options'], actions = batch['actions'], 
                                         dones = batch['dones'], rewards = rewards)
                policy_stats.save_iter(logs)
        
        if agent.on_policy:
            skill_model._update_target_te()
            paths = replay_buffer.get_recent_paths()
            original_shapes = {'obs': paths[0]['observations'].shape, 'next_obs': paths[0]['next_observations'].shape,
                               'obj_idxs': paths[0]['agent_infos']['obj_idxs'].shape, 'options': paths[0]['agent_infos']['options'].shape,
                               'rewards': paths[0]['rewards'].shape}
            
            torch_paths = None
            for batched_idx in range(0, len(paths), trainer_config.on_policy_batch):
                low, high = batched_idx, min(len(paths), batched_idx + trainer_config.on_policy_batch)
                obs = torch.concatenate([torch.from_numpy(paths[i]['observations'])
                                   for i in range(low, high)], axis=0).to(agent.device).to(torch.float32)
                next_obs = torch.concatenate([torch.from_numpy(paths[i]['next_observations'])
                                        for i in range(low, high)], axis=0).to(agent.device).to(torch.float32)
                obj_idxs = torch.concatenate([torch.from_numpy(paths[i]['agent_infos']['obj_idxs'])
                                        for i in range(low, high)], axis=0).to(agent.device)
                options = torch.concatenate([torch.from_numpy(paths[i]['agent_infos']['options']) 
                                       for i in range(low, high)], axis=0).to(agent.device)
                torch_subpaths = {'obs': obs, 'next_obs': next_obs, 'obj_idxs': obj_idxs, 'options': options}
                with torch.no_grad():
                    skill_model._update_rewards(torch_subpaths)
                    torch_subpaths.pop('cur_z'), torch_subpaths.pop('next_z')
                    for key in original_shapes.keys():
                        torch_subpaths[key] = torch_subpaths[key].reshape((high - low,) + original_shapes[key])
                if torch_paths is None:
                    torch_paths = {}
                    for key in torch_subpaths.keys():
                        torch_paths[key] = torch_subpaths[key]
                else:
                    for key in torch_paths.keys():
                        torch_paths[key] = torch.cat([torch_paths[key], torch_subpaths[key]], axis = 0)
            torch_paths['rewards'] = running_std.modify_reward(torch_paths['rewards'].cpu().numpy())
            for k in range(0, len(paths)):
                paths[k]['rewards'] = torch_paths['rewards'][k]
            rollout_buffer.update_replay_buffer(paths)
            
            j = 0
            for _ in range(trainer_config.policy_optimization_mult):
                for batch in rollout_buffer.next_batch():
                    batch = prepare_batch(batch)
                    ppo_log = agent.optimize_op(batch)
                    if ppo_log is None:
                        break
                    policy_stats.save_iter(ppo_log)
                    j += 1
            replay_buffer.delete_recent_paths()
            rollout_buffer.clear()

        if (prev_cur_step // trainer_config.log_frequency) < (cur_step // trainer_config.log_frequency):
            comet_logger.log_metrics(skill_stats.pop_statistics(), step = cur_step)
            comet_logger.log_metrics(policy_stats.pop_statistics(), 
                                      step = cur_step)
        
        agent.eval()
        if (prev_cur_step // trainer_config.eval_frequency) < (cur_step // trainer_config.eval_frequency):
            eval_metrics(eval_env, make_env_fn(seed = 0), agent, skill_model, num_random_trajectories = 48,
                            sample_processor = replay_buffer.preprocess_data, 
                            example_env_name = make_env_fn.keywords['env_name'],
                            example_env_kwargs = make_env_fn.keywords['env_kwargs'],
                            device = "cuda:0", comet_logger = comet_logger, step = cur_step)
        
        prev_cur_step = cur_step

def sample_eval_options(num_random_trajectories, skill_model):
    if skill_model.discrete:
        eye_options = np.eye(skill_model.option_size)
        random_options = []
        colors = []
        for i in range(skill_model.option_size):
            num_trajs_per_option = num_random_trajectories // skill_model.option_size + (i < num_random_trajectories % skill_model.option_size)
            for _ in range(num_trajs_per_option):
                random_options.append(eye_options[i])
                colors.append(i)
        random_options = np.array(random_options)
        colors = np.array(colors)
        num_evals = len(random_options)
        from matplotlib import cm
        cmap = 'tab10' if skill_model.option_size <= 10 else 'tab20'
        random_option_colors = []
        for i in range(num_evals):
            random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
        random_option_colors = np.array(random_option_colors)
    else:
        random_options = np.random.randn(num_random_trajectories, skill_model.option_size).astype(np.float32)
        if skill_model.unit_length:
            random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
        random_option_colors = get_option_colors(random_options * 4)
    random_options = [{'options': opt} for opt in random_options]
    return random_options, random_option_colors


def sample_video_options(skill_model):
    if skill_model.discrete:
        video_options = np.eye(skill_model.option_size)
        video_options = video_options.repeat(2, axis=0) # Num video repeats???
    else:
        if skill_model.option_size == 2:
            radius = 1. if skill_model.unit_length else 1.5
            video_options = []
            for angle in [3, 2, 1, 4]:
                video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
            video_options.append([0, 0])
            for angle in [0, 5, 6, 7]:
                video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
            video_options = np.array(video_options)
        else:
            video_options = np.random.randn(8, skill_model.option_size)
            if skill_model.unit_length:
                video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
        video_options = video_options.repeat(2, axis=0).astype(np.float32)
    return video_options


def render_ori_trajectories(options, colors, n_objects, eval_env, example_env, agent, trajectories_length = 200):
    if example_env.decoupled:
        fig, ax = plt.subplots(nrows = n_objects, ncols = n_objects)
    else:
        fig, ax = plt.subplots(nrows = 1, ncols = n_objects)
        if n_objects == 1:
            ax = np.array([ax])
        ax = np.array([ax])
    
    eval_options, random_trajectories = [], []
    for obj_i in range(n_objects):
        for elem in options:
            eval_options.append([])
            for step in range(trajectories_length):
                eval_options[-1].append({'options': elem['options'], 'obj_idxs': obj_i})
            
        random_trajectories.append(collect_trajectories(env = eval_env, agent = agent, trajectories_length = trajectories_length, 
                                                        options_and_obj_idxs = eval_options, mode = 'eval'))
        example_env.render_trajectories(random_trajectories[-1], colors, None, ax[obj_i])
    fig.canvas.draw()
    skill_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    skill_img = skill_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    for a in np.reshape(ax, -1):
        a.clear()
    plt.close(fig)
    return skill_img, random_trajectories


def render_phi_plot(skill_model, random_trajectories, eval_color, sample_processor, device):
    fig, ax = plt.subplots(1, len(random_trajectories))
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    for i, random_trajectories_per_obj in enumerate(random_trajectories):
        data = sample_processor(random_trajectories_per_obj)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(device) for ob in data['obs']]).float()
        last_obj_idx = torch.tensor([ob[-1].item() for ob in data['obj_idxs']]).to(device)
        
        option_dists = skill_model.fetch_trajectory_encoder_representation(last_obs, last_obj_idx)
        option_means = option_dists.detach().cpu().numpy()
        option_stddevs = torch.ones_like(option_dists.detach().cpu()).numpy()
        option_samples = option_dists.detach().cpu().numpy()

        draw_2d_gaussians(option_means, option_stddevs, eval_color, ax[i])
        draw_2d_gaussians(option_samples, [[0.03, 0.03]] * len(option_samples), eval_color, ax[i], fill=True, 
                          use_adaptive_axis=True)
    fig.canvas.draw()
    phi_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    phi_img = phi_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    for a in ax:
        a.clear()

    plt.close(fig)
    return phi_img


def eval_metrics(env, example_env, agent, skill_model, num_random_trajectories, 
                 sample_processor, example_env_name, example_env_kwargs, device, comet_logger, step):
    
    eval_options, eval_color = sample_eval_options(num_random_trajectories, skill_model)
    n_objects = example_env.n_obj

    # Switch policy to evaluation mode
    agent._force_use_mode_actions = True
    print('Warning! In old version _action_noise_std was setting to None seemingly does not exist.\
           Proceed with caution for new environments')
    skill_img, option_trajectories = render_ori_trajectories(options = eval_options, colors = eval_color, 
                            n_objects = n_objects, eval_env = env, example_env = example_env, agent = agent)
    comet_logger.log_image(image_data = skill_img, name = "Skill trajs", step = step)

    phi_img = render_phi_plot(skill_model = skill_model, random_trajectories = option_trajectories, 
                    eval_color=eval_color, sample_processor = sample_processor, device = device)
    comet_logger.log_image(image_data = phi_img, name = "Phi plot", step = step)

    """
    # Videos
    videos = []
    video_options = sample_video_options(skill_model)
    for obj_idx in range(n_objects):
        video_options = [{'options': opt, 'obj_idxs': obj_idx} for opt in video_options]
        video_trajectories = render_trajectories(env = example_env, agent = agent, trajectories_length = 200, 
                                                 options = video_options)
        videos.append(video_trajectories)
    """
    agent._force_use_mode_actions = False
    """
    for i, video in enumerate(videos):
        path_to_video = record_video(video, skip_frames = 2)
        comet_logger.log_video(file = path_to_video, name = 'Skill videos №{}'.format(i), step = step)
    
    comet_logger.log_metrics(example_env.calc_eval_metrics(option_trajectories[0], is_option_trajectories=True), step = step)"
    """
    example_env.close()

if __name__ == '__main__':
    matplotlib.use('Agg')
    run()
