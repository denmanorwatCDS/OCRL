import argparse, functools, sys
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
from RL.policies.ppo import PPO
from gym.vector import AsyncVectorEnv, SyncVectorEnv
from eval_utils.traj_utils import draw_2d_gaussians, render_trajectories, calc_eval_metrics
from eval_utils.eval_utils import StatisticsCalculator, monte_carlo_value_difference
from eval_utils.video_utils import record_video
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
    parser.add_argument('--env_config')
    args = parser.parse_args()

    config_folder = str(pathlib.Path(__file__).parent.resolve()) + '/configs'
    rl_config_path = config_folder + '/rl_algos/' + args.default_config
    rl_config = omegaconf.OmegaConf.load(rl_config_path)
    algo_name = rl_config.rl_algo.name.lower()

    env_config_path = config_folder + '/' + args.env_config
    env_config = omegaconf.OmegaConf.load(env_config_path)
    rl_config.merge_with(env_config)
    assert rl_config.globals.seed > 3, 'Seeds 0, 1, 2, 3 reserved for evaluation'

    return rl_config

def make_env(env_name, env_kwargs, max_path_length, seed, frame_stack, normalizer_type, 
             render_info = False):
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
        env = AntEnv(render_hw = 100, seed = seed, render_info = render_info)
        env = ExpanderWrapper(env)
        env.seed(seed = seed)
    elif env_name == 'gripper':
        from envs.mujoco.gripper_env import MultipleFetchPickAndPlaceEnv
        env = MultipleFetchPickAndPlaceEnv(seed = seed, obs_type = 'state', object_qty = env_kwargs.object_qty,
                                           object_names = env_kwargs.object_names, 
                                           render_info = render_info)
        env = ExpanderWrapper(env)
    elif env_name == 'pixel_gripper':
        from envs.mujoco.gripper_env import MultipleFetchPickAndPlaceEnv
        env = MultipleFetchPickAndPlaceEnv(seed = seed, obs_type = 'pixels', object_qty = env_kwargs.object_qty,
                                           object_names = env_kwargs.object_names,
                                           render_info = render_info)
    elif env_name == 'decoupled_gripper':
        from envs.mujoco.gripper_env import MultipleFetchPickAndPlaceEnv
        env = MultipleFetchPickAndPlaceEnv(seed = seed, obs_type = 'decoupled_state', object_qty = env_kwargs.object_qty,
                                           object_names = env_kwargs.object_names,
                                           render_info = render_info)
    elif env_name == 'decoupled_shapes':
        from envs.shapes.push_env.push import PushEnv
        env = PushEnv(seed = seed, arena_size = 2.5, render_mode = 'state', 
                      render_info = render_info, num_objects_range = [1, 1])
    elif env_name == 'easy_decoupled_shapes':
        from envs.shapes.push_env.push import PushEnv
        env = PushEnv(seed = seed, arena_size = 2.5, render_mode = 'simple_state', 
                      render_info = render_info, num_objects_range = [1, 1])
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
    env = make_seeded_env(seed = config.globals.seed, render_info = False)
    
    # TODO change shape
    if config.rl_algo.name == 'SAC':
        rl_algo = SAC(name = 'SAC', obs_length = env.observation_space.shape[1], task_length = config.skill.dim_option, 
                  action_length = env.action_space.shape[0], actor_config = config.rl_algo.policy, obj_qty = env.n_obj,
                  critic_config = config.rl_algo.critics, pooler_config = config.rl_algo.slot_pooler,
                  alpha = config.rl_algo.alpha.value, tau = config.rl_algo.tau, scale_reward = config.rl_algo.scale_reward,
                  env_spec = env, target_coef = config.rl_algo.target_coef, device = config.globals.device,
                  discount = config.rl_algo.discount, lr = config.rl_algo.lr, wd = config.rl_algo.wd)
        
    elif config.rl_algo.name == 'PPO':
        rl_algo = PPO(name = 'PPO', obs_length = env.observation_space.shape[1], task_length = config.skill.dim_option,
                      action_length = env.action_space.shape[0], actor_config = config.rl_algo.policy,
                      critic_config = config.rl_algo.value, pooler_config = config.rl_algo.slot_pooler,
                      actor_lr = config.rl_algo.policy.lr, critic_lr = config.rl_algo.value.lr, 
                      pooler_lr = config.rl_algo.slot_pooler.lr, actor_wd = config.rl_algo.policy.wd, 
                      critic_wd = config.rl_algo.value.wd, pooler_wd = config.rl_algo.slot_pooler.wd,
                      clip_coef = config.rl_algo.clip_coef, ent_coef = config.rl_algo.ent_coef,
                      vf_coef = config.rl_algo.vf_coef, normalize_advantage = config.rl_algo.norm_adv,
                      max_grad_norm = config.rl_algo.max_grad_norm, device = config.globals.device,
                      target_kl = config.rl_algo.target_kl)

    replay_buffer = PathBuffer(capacity_in_transitions = int(config.replay_buffer.common.max_transitions), 
                               batch_size = config.replay_buffer.common.batch_size, pixel_keys = {}, 
                               discount = config.replay_buffer.common.discount, 
                               gae_lambda = config.replay_buffer.discount, seed = config.globals.seed,)
                               #path_to_perfect_buffer = '/home/denis/Work/METRA_simplified/perfect_buffer.pickle')

    metra = METRA(obs_length = env.observation_space.shape[1], pooler_config = config.skill.slot_pooler, 
                  traj_encoder_config = config.skill.trajectory_encoder,
                  lr = config.skill.lr, wd = config.skill.wd,
                  dual_lam = config.skill.dual_lam,
                  option_size = config.skill.dim_option, discrete = config.skill.discrete, 
                  unit_length = config.skill.unit_length, device = config.globals.device,
                  dual_slack = config.skill.dual_slack)
    env.close()
    
    train_cycle(config.trainer_args, agent = rl_algo, skill_model = metra, replay_buffer = replay_buffer,
                make_env_fn = make_seeded_env, seed = config.globals.seed, comet_logger = exp)

def collect_train_trajectories(env, agent, skill_model, 
                               trajectories_length, skills_per_traj = None, n_objects = None, trajectories_qty = None):
    options, obj_idxs = skill_model.sample_options_and_obj_idxs(batch_size = trajectories_qty, 
                                                                traj_len = trajectories_length,
                                                                skills_per_traj = skills_per_traj, 
                                                                n_objects = n_objects)
    return collect_trajectories(env, agent, mode = 'train', options = options, obj_idxs = obj_idxs)

def collect_eval_trajectories(env, agent, options, obj_idxs):
    return collect_trajectories(env, agent, mode = 'eval', 
                                options = options, obj_idxs = obj_idxs)

def collect_trajectories(env, agent, mode, options = None, obj_idxs = None):
    trajectories_qty, trajectories_length = options.shape[:2]
    
    env_qty = len(env.env_fns)
    if isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
        pseudoepisodes = math.ceil(trajectories_qty / len(env.env_fns))
    else:
        pseudoepisodes = trajectories_qty
    
    cache = []
    for pseudoepisode in range(pseudoepisodes):
        prev_obs, prev_dones = env.reset(), np.full((env_qty,), fill_value=False)
        for i in range(trajectories_length):
            b_opt = options[pseudoepisode * env_qty: (pseudoepisode + 1) * env_qty, i]
            b_obj_idxs = obj_idxs[pseudoepisode * env_qty: (pseudoepisode + 1) * env_qty, i]
            action, action_info = agent.get_actions(prev_obs, b_opt, b_obj_idxs)
            next_obs, rewards, dones, env_infos = env.step(action)
            next_obs = np.transpose(next_obs, [0, 3, 1, 2]) if len(next_obs.shape) == 4 else next_obs
            data = {'observations': prev_obs, 'next_observations': next_obs, 'rewards': rewards, 
                    'dones': dones, 'actions': action, 'options': b_opt, 'obj_idxs': b_obj_idxs}
            if mode == 'train':
                data.update(**action_info)
            elif mode == 'eval':
                env_infos = {key: np.stack([env_infos[i][key] for i in range(len(env_infos))], axis=0) for key in env_infos[0].keys()}
                data.update(**env_infos)

            if pseudoepisode == 0:
                cache.append(data)
            else:
                cache[i] = {key: np.concatenate([cache[i][key], data[key]], axis = 0) for key in cache[i].keys()}
            prev_obs = next_obs
            prev_dones = np.logical_or(prev_dones, dones)
    
    tensors_by_key = {}
    for key in cache[-1].keys():
        tensors_by_key[key] = np.stack([cache[i][key] for i in range(len(cache))], axis = 1)

    return tensors_by_key

def prepare_batch(batch, device = 'cuda'):
    data = {}
    for key, value in batch.items():
        data[key] = torch.from_numpy(value).to(device)
    return data

def train_cycle(trainer_config, agent, skill_model, replay_buffer, make_env_fn, seed, 
                comet_logger):
    n_objects = make_env_fn(seed = 0).n_obj
    env = AsyncVectorEnv([lambda: make_env_fn(seed = (seed + i)) for i in range(trainer_config.n_parallel)], context='spawn')
    
    prev_cur_step, cur_step = 0, 0
    for i in range(trainer_config.n_epochs):
        agent.eval()
        trajs = collect_train_trajectories(env = env, agent = agent, skill_model = skill_model, 
                                           skills_per_traj = trainer_config.skills_per_trajectory, n_objects = n_objects,
                                           trajectories_qty = trainer_config.traj_batch_size, 
                                           trajectories_length = trainer_config.max_path_length)
        prev_cur_step = cur_step
        for traj in trajs['dones']:
            cur_step += len(traj)
        replay_buffer.update_replay_buffer(trajs)
        if (replay_buffer.n_transitions_stored < trainer_config.transitions_before_training):
            continue
        
        agent.train()
        skill_stats = StatisticsCalculator('skill')
        policy_stats = StatisticsCalculator('policy')

        for i in range(trainer_config.trans_optimization_epochs):
            batch = replay_buffer.sample_transitions()
            batch = prepare_batch(batch)
            logs, rewards = skill_model.train_components(observations = batch['observations'], 
                                                         next_observations = batch['next_observations'],
                                                         options = batch['options'],
                                                         obj_idxs = batch['obj_idxs'])
            skill_stats.save_iter(logs)
            if not agent.on_policy:
                logs = agent.optimize_op(observations = batch['observations'], next_observations = batch['next_observations'], 
                                         obj_idxs = batch['obj_idxs'], options = batch['options'], 
                                         actions = batch['actions'], dones = batch['dones'], rewards = rewards)
                policy_stats.save_iter(logs)
        
        if agent.on_policy:
            trajs['rewards'] = skill_model.calculate_rewards(observations = trajs['observations'], 
                                                    next_observations = trajs['next_observations'],
                                                    options = trajs['options'], obj_idxs = trajs['obj_idxs'])
            trajs['values'] = agent.get_critic_value(trajs['observations'], trajs['options'], trajs['obj_idxs'])
            trajs['next_values'] = agent.get_critic_value(trajs['next_observations'], trajs['options'], trajs['obj_idxs'])
            replay_buffer.update_rollout_buffer(trajs)
            
            for _ in range(trainer_config.policy_optimization_epochs):
                for batch in replay_buffer.get_rollout_iterator(trainer_config.policy_batch_size):
                    batch = prepare_batch(batch)
                    ppo_log = agent.optimize_op(observations = batch['observations'], 
                                                obj_idxs = batch['obj_idxs'], options = batch['options'], 
                                                actions = batch['actions'], pre_tanh_actions = batch['pre_tanh_value'], 
                                                old_logprobs = batch['log_prob'], 
                                                advantages = batch['advantages'], returns = batch['returns'])
                    if ppo_log is None:
                        break
                    policy_stats.save_iter(ppo_log)
        
        if (prev_cur_step // trainer_config.log_frequency) < (cur_step // trainer_config.log_frequency):
            comet_logger.log_metrics(skill_stats.pop_statistics(), step = cur_step)
            comet_logger.log_metrics(policy_stats.pop_statistics(), step = cur_step)
        
        agent.eval()
        if (prev_cur_step // trainer_config.eval_frequency) < (cur_step // trainer_config.eval_frequency):
            eval_metrics(make_env_fn, agent, skill_model, 
                         num_random_trajectories = 48, traj_length = trainer_config.max_path_length,
                         gamma = replay_buffer.discount, sample_processor = replay_buffer.preprocess_data,
                         device = "cuda:0", comet_logger = comet_logger, step = cur_step)
        
        prev_cur_step = cur_step

def render_ori_trajectories(options, colors, n_slots, n_objects, eval_env_maker, agent):
    fig, axs = plt.subplots(nrows = n_slots, ncols = n_objects)
    fig.set_size_inches(15, 15)
    if isinstance(axs, matplotlib.axes._axes.Axes):
        axs = [[axs]]
    
    random_trajectories = []
    for slot_i in range(n_slots):
        obj_idxs = np.zeros(options.shape[:-1], dtype = np.int32) + slot_i
        eval_env = eval_env_maker()
        obj_trajectories = collect_eval_trajectories(env = eval_env, agent = agent, 
                                                     options = options, obj_idxs = obj_idxs)
        for obj_i in range(n_objects):
            coordinates = obj_trajectories['coordinates'][:, :, obj_i]
            last_coordinate = obj_trajectories['next_coordinates'][:, -1:, obj_i]
            # Pad coordinates with last observed position, whilst it is guaranteed that options are consistent
            # across trajectories, thus simply copy last options color one additional time
            tmp_coordinates = np.concatenate([coordinates, last_coordinate], axis = 1)
            axs[slot_i][obj_i].set_title(f'Slot №{slot_i} Object №{obj_i}')
            render_trajectories(tmp_coordinates, colors, None, axs[slot_i][obj_i])
        eval_env.close()
        random_trajectories.append(obj_trajectories)
    fig.canvas.draw()
    skill_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype = np.uint8)
    skill_img = skill_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    for sub_axs in axs:
        for ax in sub_axs:
            ax.clear()
    plt.close(fig)
    return skill_img, random_trajectories

def render_phi_plot(skill_model, random_trajectories, eval_color, sample_processor, device):
    fig, axs = plt.subplots(1, len(random_trajectories))
    fig.set_size_inches(15, 15)
    if isinstance(axs, matplotlib.axes._axes.Axes):
        axs = [axs]

    for i, random_trajectories_per_slot in enumerate(random_trajectories):
        data = sample_processor(random_trajectories_per_slot)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(device) for ob in data['observations']]).float()
        last_obj_idx = torch.tensor([ob[-1].item() for ob in data['obj_idxs']]).to(device)
        
        means, stds, samples = skill_model.fetch_encoder_representation(last_obs, last_obj_idx)
        axs[i].set_title(f'Object №{i}')
        draw_2d_gaussians(means, stds, eval_color, axs[i])
        draw_2d_gaussians(samples, [[0.03, 0.03]] * len(samples), eval_color, axs[i], fill=True, 
                          use_adaptive_axis=True)
    fig.canvas.draw()
    phi_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    phi_img = phi_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    for ax in axs:
        ax.clear()

    plt.close(fig)
    return phi_img

def eval_metrics(make_env_fn, agent, skill_model, num_random_trajectories, traj_length, gamma,
                 sample_processor, device, comet_logger, step):
    example_env = make_env_fn(seed = 0)
    traj_env_maker = lambda: AsyncVectorEnv([lambda: make_env_fn(seed = i) for i in range(4)], context='spawn')
    eval_options, eval_color = skill_model.sample_eval_options(num_random_trajectories, traj_length)
    n_objects = example_env.n_obj
    n_slots = n_objects

    # Switch policy to evaluation mode
    agent._force_use_mode_actions = True
    print('Warning! In old version _action_noise_std was setting to None seemingly does not exist.\
           Proceed with caution for new environments')
    skill_img, option_trajectories = render_ori_trajectories(options = eval_options, colors = eval_color, 
                                                             n_slots = n_slots, n_objects = n_objects, eval_env_maker = traj_env_maker, 
                                                             agent = agent)
    comet_logger.log_image(image_data = skill_img, name = "Skill trajs", step = step)

    phi_img = render_phi_plot(skill_model = skill_model, random_trajectories = option_trajectories, 
                             eval_color = eval_color, sample_processor = sample_processor, device = device)
    comet_logger.log_image(image_data = phi_img, name = "Phi plot", step = step)

    # Videos
    videos = []
    video_options = skill_model.sample_fixated_options(traj_length)
    for slot_idx in range(n_slots):
        video_env = AsyncVectorEnv([lambda: make_env_fn(seed = i, render_info = True) for i in range(2)], context='spawn')
        obj_idxs = np.zeros(video_options.shape[:-1], dtype = np.int32) + slot_idx
        video_trajectories = collect_eval_trajectories(env = video_env, agent = agent, options = video_options, obj_idxs = obj_idxs)
        videos.append(video_trajectories['render'])
        video_env.close()

    agent._force_use_mode_actions = False
    for i, skills_videos in enumerate(videos):
        path_to_video = record_video(skills_videos, skip_frames = 2)
        comet_logger.log_video(file = path_to_video, name = f'Slot №{i}'.format(i), step = step)
    
    mets = {}
    if n_objects == 1:
        mets.update(calc_eval_metrics(option_trajectories[0]['coordinates'], example_env.env_discretizer()))
    else:
        for obj_idx in range(n_objects):
            mets.update(calc_eval_metrics(option_trajectories[obj_idx]['coordinates'][:, :, obj_idx], 
                                          example_env.env_discretizer(), 
                                          prefix = f'Object№{obj_idx}'))
    for obj_idx in range(n_objects):
        rewards = skill_model.calculate_rewards(observations = option_trajectories[obj_idx]['observations'], 
                                                next_observations = option_trajectories[obj_idx]['next_observations'],
                                                options = option_trajectories[obj_idx]['options'], 
                                                obj_idxs = option_trajectories[obj_idx]['obj_idxs'])
        values = agent.inference_value(observations = option_trajectories[obj_idx]['observations'],
                                       actions = option_trajectories[obj_idx]['actions'],
                                       options = option_trajectories[obj_idx]['options'],
                                       obj_idxs = option_trajectories[obj_idx]['obj_idxs'])

        mc_value_differences = monte_carlo_value_difference(rewards, gamma = gamma)
        predicted_value_differences = values - values[:, -2: -1].repeat(values.shape[1], axis = 1) *\
            (np.fliplr(np.cumprod(np.ones(values.shape) * gamma, axis = 1)) / gamma)
        mets.update({f'Truncated_returns№{obj_idx}': np.mean(mc_value_differences)})
        mets.update({f'Predicted_truncated_returns№{obj_idx}': np.mean(predicted_value_differences)})
        mets.update({f'Values_№{obj_idx}': np.mean(values)})
        mets.update({f'Mean_error№{obj_idx}': np.mean(mc_value_differences - predicted_value_differences)})
        mets.update({f'Mean_absolute_error№{obj_idx}': np.mean(np.abs(mc_value_differences - predicted_value_differences))})
    mets = {f'val/{key}': val for key, val in mets.items()}
    comet_logger.log_metrics(mets, step = step)
    example_env.close(), video_env.close()
    del example_env, video_env

if __name__ == '__main__':
    matplotlib.use('Agg')
    run()