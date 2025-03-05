import argparse, datetime, functools, sys, os, tempfile
import comet_ml
import numpy as np
import omegaconf
import pathlib
import torch
from envs.utils.consistent_normalized_env import consistent_normalize, get_normalizer_preset
from networks.distribution_networks import GaussianMLPModule, GaussianMLPIndependentStdModule, GaussianMLPTwoHeadedModule
from utils.distributions.tanh import TanhNormal
from utils.weight_initializer.xavier_init import xavier_normal
from RL.policies.policy import Policy
from ReplayBuffers.path_replay_buffer import PathBuffer
from networks.mlp import ContinuousMLPQFunction
from skill_discovery.METRA.metra import METRA
from RL.algos.sac import SAC
from networks.cnn import Encoder, ConcatEncoder, WithEncoder
from networks.parameter import ParameterModule
from gym.vector import AsyncVectorEnv
from copy import deepcopy
from eval_utils.eval_utils import get_option_colors, draw_2d_gaussians, record_video
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
    parser.add_argument('--config')
    args = parser.parse_args()
    config_folder = str(pathlib.Path(__file__).parent.resolve()) + '/configs'
    config = config_folder + '/' + 'default.yaml'
    config = omegaconf.OmegaConf.load(config)
    if args.config != 'default':
        subconfig = config_folder + '/' + args.config
        subconfig = omegaconf.OmegaConf.load(subconfig)
        config.merge_with(subconfig)

    return config


def make_env(env_name, encoder, max_path_length, seed, frame_stack, normalizer_type):
    if env_name == 'maze':
        from envs.maze_env import MazeEnv
        env = MazeEnv(
            max_path_length=max_path_length,
            action_range=0.2,
        )
    elif env_name == 'half_cheetah':
        from envs.mujoco.half_cheetah_env import HalfCheetahEnv
        env = HalfCheetahEnv(render_hw=100)
        env.seed(seed = seed)
    elif env_name == 'ant':
        from envs.mujoco.ant_env import AntEnv
        env = AntEnv(render_hw=100)
        env.seed(seed = seed)
    elif env_name.startswith('dmc'):
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
        assert encoder in ['cnn']  # Only support pixel-based environments
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
        assert encoder in ['cnn']  # Only support pixel-based environments
        assert seed is None, 'For some strange reason, this environment does not have any seed...'
        env = MyKitchenEnv(log_per_goal=True)
    else:
        raise NotImplementedError

    if frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
        env = FrameStackWrapper(env, frame_stack)

    normalizer_kwargs = {}

    if normalizer_type == 'off':
        env = consistent_normalize(env, normalize_obs = False, **normalizer_kwargs)
    elif normalizer_type == 'squashed':
        env = consistent_normalize(env, flatten_obs = False, normalize_obs = True, 
                                   mean = 255. / 2, std = 255. / 2)
    elif normalizer_type == 'preset':
        normalizer_name = env_name
        normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        env = consistent_normalize(env, normalize_obs = True, mean = normalizer_mean, std = normalizer_std, 
                                   **normalizer_kwargs)

    return env


def fetch_activation(activation_name):
    assert activation_name in ['relu', 'tanh'], 'Only relu or tanh are supported as activations'
    if activation_name == 'relu':
        return torch.relu
    elif activation_name == 'tanh':
        return torch.tanh


def fetch_dist_type(class_name):
    assert class_name in ['TanhNormal']
    if class_name == 'TanhNormal':
        return TanhNormal


def build_policy_net(env, encoder, policy_net_config):
    obs_dim, action_dim = env.spec.observation_space.flat_dim, env.spec.action_space.flat_dim
    
    if encoder == 'cnn':
        example_ob = np.transpose(env.reset(), [2, 0, 1])
        depth = example_ob.shape[0]

        encoder = Encoder(pixel_depth = depth, obs_key = 'obs', concat_keys = ['option'])
        example_ob = {'obs': torch.as_tensor(example_ob).float().unsqueeze(0), 
                      'option': torch.zeros((1, policy_net_config.dim_option))}
        module_obs_dim = encoder(example_ob).shape[-1]
    elif encoder == 'concat':
        encoder = ConcatEncoder(obs_key = 'obs', concat_keys = ['option'])
        module_obs_dim = obs_dim + policy_net_config.dim_option
    
    policy_module = GaussianMLPTwoHeadedModule(input_dim = module_obs_dim, output_dim = action_dim, 
                                               hidden_sizes = policy_net_config.hidden_sizes, 
                                               layer_normalization = policy_net_config.layer_normalization,
                                               hidden_nonlinearity = fetch_activation(policy_net_config.nonlinearity), 
                                               max_std = np.exp(policy_net_config.distribution.max_logstd),
                                               init_std = np.exp(policy_net_config.distribution.starting_logstd),
                                               normal_distribution_cls = fetch_dist_type(policy_net_config.distribution.type),
                                               output_w_init = functools.partial(xavier_normal, gain=1.))
    
    policy_module = WithEncoder(encoder = encoder, module = policy_module)
    return Policy(name = policy_net_config.name, module = policy_module)


def build_trajectory_encoder(env, encoder, trajectory_net_config):
    obs_dim = env.spec.observation_space.flat_dim

    if encoder == 'cnn':
        example_ob = np.transpose(env.reset(), [2, 0, 1])
        depth = example_ob.shape[0]

        encoder = Encoder(pixel_depth = depth, obs_key = 'obs', concat_keys = [], 
                          spectral_normalization=trajectory_net_config.spectral_normalization)
        example_ob = {'obs': torch.as_tensor(example_ob).float().unsqueeze(0)}
        module_obs_dim = encoder(example_ob).shape[-1]
    elif encoder == 'concat':
        encoder = ConcatEncoder(obs_key = 'obs', concat_keys = [])
        module_obs_dim = obs_dim

    traj_encoder = GaussianMLPIndependentStdModule(hidden_sizes = trajectory_net_config.hidden_sizes, 
                                                   std_hidden_sizes = trajectory_net_config.hidden_sizes,
                                                   hidden_nonlinearity = fetch_activation(trajectory_net_config.nonlinearity),
                                                   std_hidden_nonlinearity = fetch_activation(trajectory_net_config.nonlinearity),
                                                   std_hidden_w_init = torch.nn.init.xavier_uniform_, 
                                                   std_output_w_init = torch.nn.init.xavier_uniform_,
                                                   hidden_w_init = torch.nn.init.xavier_uniform_, 
                                                   output_w_init = torch.nn.init.xavier_uniform_,
                                                   init_std = trajectory_net_config.init_std,
                                                   min_std = trajectory_net_config.min_std,
                                                   max_std = trajectory_net_config.max_std,
                                                   std_parameterization = trajectory_net_config.std_parameterization,
                                                   bias = trajectory_net_config.bias,
                                                   input_dim = module_obs_dim, 
                                                   output_dim = trajectory_net_config.dim_option,
                                                   spectral_normalization = trajectory_net_config.spectral_normalization)
    
    traj_encoder = WithEncoder(module = traj_encoder, encoder = encoder)
    return traj_encoder


def build_q_net(env, encoder, q_net_config):
    obs_dim, action_dim = env.spec.observation_space.flat_dim, env.spec.action_space.flat_dim

    if encoder == 'cnn':
        example_ob = np.transpose(env.reset(), [2, 0, 1])
        depth = example_ob.shape[0]

        encoder = Encoder(pixel_depth = depth, obs_key = 'obs', concat_keys = ['option'])
        example_ob = {'obs': torch.as_tensor(example_ob).float().unsqueeze(0), 
                      'option': torch.zeros((1, q_net_config.dim_option))}
        module_obs_dim = encoder(example_ob).shape[-1]
    elif encoder == 'concat':
        encoder = ConcatEncoder(obs_key = 'obs', concat_keys = ['option'])
        module_obs_dim = obs_dim + q_net_config.dim_option

    q = ContinuousMLPQFunction(obs_dim = module_obs_dim, action_dim = action_dim, 
                               hidden_sizes = q_net_config.hidden_sizes,
                               hidden_nonlinearity = fetch_activation(q_net_config.nonlinearity))
    
    q = WithEncoder(module = q, encoder = encoder)
    return q
    

def build_dist_predictor(dual_dist_name):
    assert dual_dist_name in ['l2', 'one'], 's2_from_s and other distances are not supported. Yet...'
    return None


def build_skill_dynamics(skill_algo_name):
    assert skill_algo_name == 'METRA', 'dads and other algos are not supported. Yet...'
    return None

def run():
    config = fetch_config()
    g_start_time = int(datetime.datetime.now().timestamp())

    exp = comet_ml.start(project_name = 'metra')
    exp.log_parameters(config)

    print('ARGS: ' + str(config))
    if config.globals.n_thread is not None:
        torch.set_num_threads(config.globals.n_thread)

    set_seed(config.globals.seed)
    make_seeded_env = functools.partial(make_env, env_name = config.env.name, 
                                        encoder = config.globals.encoder,
                                        max_path_length = config.env.max_path_length,
                                        frame_stack = config.env.frame_stack, 
                                        normalizer_type = config.env.normalizer_type)
    env = make_seeded_env(seed = config.globals.seed)

    option_policy = build_policy_net(env, encoder = config.globals.encoder, 
                                     policy_net_config = config.rl_algo.policy)

    traj_encoder = build_trajectory_encoder(env, encoder = config.globals.encoder,
                                            trajectory_net_config = config.skill.trajectory_encoder)
    dist_predictor = None
    skill_dynamics = None

    dual_lam = ParameterModule(torch.Tensor([np.log(config.skill.dual_lam)]))

    optimizers = {
        'option_policy': torch.optim.Adam([
            {'params': option_policy.parameters(), 'lr': config.rl_algo.policy.lr},
        ]),
        'traj_encoder': torch.optim.Adam([
            {'params': traj_encoder.parameters(), 'lr': config.skill.trajectory_encoder.lr},
        ]),
        'dual_lam': torch.optim.Adam([
            {'params': dual_lam.parameters(), 'lr': config.skill.dual_lr},
        ]),
    }

    pixel_keys = ['obs', 'next_obs'] if config.env.name in ['dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid'] else []
    replay_buffer = PathBuffer(capacity_in_transitions = int(config.replay_buffer.max_transitions), 
                               pixel_keys = pixel_keys, discount = config.replay_buffer.discount)
    
    qf1, qf2 = build_q_net(env, encoder = config.globals.encoder, q_net_config = config.rl_algo.critics),\
        build_q_net(env, encoder = config.globals.encoder, q_net_config = config.rl_algo.critics)
        
    log_alpha = ParameterModule(torch.Tensor([np.log(config.rl_algo.alpha.value)]))
    optimizers.update({
        'qf': torch.optim.Adam([
            {'params': list(qf1.parameters()) + list(qf2.parameters()), 'lr': config.rl_algo.critics.lr},
        ]),
        'log_alpha': torch.optim.Adam([
            {'params': log_alpha.parameters(), 'lr': config.rl_algo.alpha.lr},
        ])
    })

    metra = METRA(traj_encoder = traj_encoder, dual_lam = dual_lam, 
                  optimizers = {key: optimizers[key] for key in ['traj_encoder', 'dual_lam']}, 
                  dim_option = config.skill.dim_option, discrete = config.skill.discrete, 
                  unit_length = config.skill.unit_length, device = config.globals.device, 
                  dual_reg = config.skill.dual_reg, dual_slack = config.skill.dual_slack, 
                  dual_dist = config.skill.dual_dist)
    
    sac_algo = SAC(qf1 = qf1, qf2 = qf2, log_alpha = log_alpha, tau = config.rl_algo.tau, 
                   scale_reward = config.rl_algo.scale_reward, target_coef = config.rl_algo.target_coef,
                   option_policy = option_policy, device = config.globals.device,
                   optimizers = {key: optimizers[key] for key in ['option_policy', 'qf', 'log_alpha']},
                   discount = config.rl_algo.discount, env_spec = env)
    
    train_cycle(config.trainer_args, agent = sac_algo, skill_model = metra, replay_buffer=replay_buffer,
                make_env_fn = make_seeded_env, seed = config.globals.seed, comet_logger = exp)


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
            elif isinstance(infos[key], (np.float32, float)):
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


def collect_trajectories(env, agent, trajectories_length, trajectories_qty = None, skill_model = None, 
                         options = None, render = False):
    if options is None:
        options = skill_model._get_train_trajectories_kwargs(trajectories_qty)
    else:
        assert (trajectories_qty is None) and (skill_model is None), """It is expected, that when using options, 
        there are one trajectory per each option, and there is no need to sample new options"""
        trajectories_qty = len(options)

    env_qty = len(env.env_fns)
    assert trajectories_qty % len(env.env_fns) == 0, 'Not integer division'
    if isinstance(env, AsyncVectorEnv):
        pseudoepisodes = trajectories_qty // len(env.env_fns)
    else:
        pseudoepisodes = trajectories_qty
    
    generated_trajectories = [{} for i in range(trajectories_qty)]
    for pseudoepisode in range(pseudoepisodes):
        cur_options = np.array([options[pseudoepisode * env_qty + i]['option'] for i in range(env_qty)])
        prev_obs = env.reset()
        prev_obs = np.transpose(prev_obs, [0, 3, 1, 2]) if len(prev_obs.shape) == 4 else prev_obs
        prev_dones = np.full((env_qty,), fill_value=False)
        for i in range(trajectories_length):
            obs_and_option = {'obs': prev_obs, 'option': cur_options}
            action, action_info = agent.policy['option_policy'].get_actions(obs_and_option)
            next_obs, rewards, dones, env_infos = env.step(action) # TODO pass render here. When switched from pointmaze to ant.
            next_obs = np.transpose(next_obs, [0, 3, 1, 2]) if len(next_obs.shape) == 4 else next_obs
            if (i == trajectories_length - 1):
                dones = np.full((env_qty,), fill_value = True)
            for i, done in enumerate(prev_dones):
                if not done:
                    update_traj_with_array(generated_trajectories[i + pseudoepisode * env_qty], prev_obs[i: i+1], 
                                           'observations')
                    update_traj_with_array(generated_trajectories[i + pseudoepisode * env_qty], next_obs[i: i+1], 
                                           'next_observations')
                    update_traj_with_array(generated_trajectories[i + pseudoepisode * env_qty], action[i: i+1], 
                                           'actions')
                    update_traj_with_array(generated_trajectories[i + pseudoepisode * env_qty], rewards[i: i+1], 
                                           'rewards')
                    update_traj_with_info(generated_trajectories[i + pseudoepisode * env_qty], env_infos[i], 'env_infos')

                    agent_info = deepcopy({key: action_info[key][i] for key in action_info.keys()})
                    agent_info.update({'option': cur_options[i]})
                    update_traj_with_info(generated_trajectories[i + pseudoepisode * env_qty], agent_info, 'agent_infos')
                    update_traj_with_array(generated_trajectories[i + pseudoepisode * env_qty], dones[i: i+1], 
                                           'dones')
            prev_obs = next_obs
            prev_dones = np.logical_or(prev_dones, dones)
    return generated_trajectories

def prepare_batch(batch, device = 'cuda'):
    data = {}
    for key, value in batch.items():
        if value.shape[1] == 1 and 'option' not in key:
            value = np.squeeze(value, axis=1)
        data[key] = torch.from_numpy(value).float().to(device)
    data['obs'] = {'obs': data['obs'], 'option': data['options']}
    data['next_obs'] = {'obs': data['next_obs'], 'option': data['next_options']}
    return data

def train_cycle(trainer_config, agent, skill_model, replay_buffer, make_env_fn, seed, 
                comet_logger):
    env = AsyncVectorEnv([lambda: make_env_fn(seed = (seed + i)) for i in range(trainer_config.n_parallel)],
                        context = "spawn")
    eval_env = AsyncVectorEnv([lambda: make_env_fn(seed = (seed + i)) for i in range(trainer_config.n_parallel)],
                                context = "spawn")
    cur_step = 0
    for i in range(trainer_config.n_epochs):
        trajs = collect_trajectories(env = env, agent = agent, skill_model = skill_model, 
                            trajectories_qty = trainer_config.traj_batch_size, 
                            trajectories_length = trainer_config.max_path_length)
        for traj in trajs:
            cur_step += len(traj)
        replay_buffer.update_replay_buffer(trajs)
        if replay_buffer.n_transitions_stored < trainer_config.transitions_before_training:
            continue
        for _ in range(trainer_config.trans_optimization_epochs):
            batch = replay_buffer.sample_transitions(batch_size = trainer_config.batch_size)
            batch = prepare_batch(batch)
            logs, modified_batch = skill_model.train_components(batch)
            logs.update(agent.optimize_op(modified_batch))
        if i % 50 == 0:
            comet_logger.log_metrics(logs, step = cur_step)
        if i % 250 == 0:
            eval_metrics(eval_env, make_env_fn(seed = 0), agent, skill_model, num_random_trajectories = 48,
                            sample_processor = replay_buffer.preprocess_data, example_env_name = make_env_fn.keywords['env_name'],
                            device = "cuda:0", comet_logger = comet_logger, step = cur_step)


def eval_metrics(env, example_env, agent, skill_model, num_random_trajectories, 
                 sample_processor, example_env_name, device, comet_logger, step):
    if skill_model.discrete:
        eye_options = np.eye(skill_model.dim_option)
        random_options = []
        colors = []
        for i in range(skill_model.dim_option):
            num_trajs_per_option = num_random_trajectories // skill_model.dim_option + (i < num_random_trajectories % skill_model.dim_option)
            for _ in range(num_trajs_per_option):
                random_options.append(eye_options[i])
                colors.append(i)
        random_options = np.array(random_options)
        colors = np.array(colors)
        num_evals = len(random_options)
        from matplotlib import cm
        cmap = 'tab10' if skill_model.dim_option <= 10 else 'tab20'
        random_option_colors = []
        for i in range(num_evals):
            random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
        random_option_colors = np.array(random_option_colors)
    else:
        random_options = np.random.randn(num_random_trajectories, skill_model.dim_option)
        if skill_model.unit_length:
            random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
        random_option_colors = get_option_colors(random_options * 4)
    random_options = [{'option': opt} for opt in random_options]
    # Switch policy to evaluation mode
    agent.option_policy._force_use_mode_actions = True
    print('Warning! In old version _action_noise_std was setting to None seemingly does not exist. Proceed with caution for new environments')
    random_trajectories = collect_trajectories(env = env, agent = agent, trajectories_length = 200, 
                                               options = random_options)
    
    fig, ax = plt.subplots()
    example_env.render_trajectories(random_trajectories, random_option_colors, None, ax)
    fig.canvas.draw()
    skill_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    skill_img = skill_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    ax.clear(), plt.close(fig)
    comet_logger.log_image(image_data = skill_img, name = "Skill trajs", step = step)

    data = sample_processor(random_trajectories)
    last_obs = torch.stack([torch.from_numpy(ob[-1]).to(device) for ob in data['obs']]).float()
    last_obs = {'obs': last_obs}
    option_dists = skill_model.traj_encoder(last_obs)

    option_means = option_dists.mean.detach().cpu().numpy()
    option_stddevs = torch.ones_like(option_dists.stddev.detach().cpu()).numpy()
    option_samples = option_dists.mean.detach().cpu().numpy()

    option_colors = random_option_colors

    fig, ax = plt.subplots()
    draw_2d_gaussians(option_means, option_stddevs, option_colors, ax)
    draw_2d_gaussians(option_samples, [[0.03, 0.03]] * len(option_samples), option_colors, ax, fill=True, 
                      use_adaptive_axis=True)
    fig.canvas.draw()
    phi_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    phi_img = phi_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    ax.clear(), plt.close(fig)
    comet_logger.log_image(image_data = phi_img, name = "Phi plot", step = step)
    agent.option_policy._force_use_mode_actions = False

    # Videos
    if skill_model.discrete:
        video_options = np.eye(skill_model.dim_option)
        video_options = video_options.repeat(2, axis=0) # Num video repeats???
    else:
        if skill_model.dim_option == 2:
            radius = 1. if skill_model.unit_length else 1.5
            video_options = []
            for angle in [3, 2, 1, 4]:
                video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
            video_options.append([0, 0])
            for angle in [0, 5, 6]:
                video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
            video_options = np.array(video_options)
        else:
            video_options = np.random.randn(8, skill_model.dim_option)
            if skill_model.unit_length:
                video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
        video_options = video_options.repeat(2, axis=0)
    video_options = [{'option': opt} for opt in video_options]
    video_trajectories = collect_trajectories(env = env, agent = agent, trajectories_length = 200, 
                                              options = video_options, render = True)
    video_trajectories = fetch_frames(video_trajectories, example_env = example_env, env_name = example_env_name)
    path_to_video = record_video(video_trajectories, skip_frames = 2)
    comet_logger.log_video(file = path_to_video, name = 'Skill videos', step = step)
    
    comet_logger.log_metrics(example_env.calc_eval_metrics(random_trajectories, is_option_trajectories=True), step = step)
    example_env.close()

def fetch_frames(trajectories, example_env, env_name):
    states = [trajectories[i]['observations'] for i in range(len(trajectories))]
    actions = [trajectories[i]['actions'] for i in range(len(trajectories))]
    if env_name in ['dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid']:
        return (np.transpose(np.array(states), [0, 1, 3, 4, 2]) * (255 / 2) + (255 / 2)).astype(np.uint8)
    unwrapped_example_env = example_env.unwrapped

    video = []
    if env_name == 'ant':
        for i, traj in enumerate(states):
            video.append([])
            for j, step in enumerate(traj):
                if j == 0:
                    unwrapped_example_env.reset()
                unwrapped_example_env.step(actions[i][j])
                video[-1].append(unwrapped_example_env.render(mode = 'rgb_array', width = 100, height = 100))
    return np.array(video)


if __name__ == '__main__':
    matplotlib.use('Agg')
    run()
