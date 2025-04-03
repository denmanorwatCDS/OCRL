import argparse, datetime, functools, sys
import comet_ml
import numpy as np
import omegaconf
import pathlib
import torch
import math

from envs.utils.consistent_normalized_env import consistent_normalize, get_normalizer_preset
from envs.mujoco.obs_wrapper import ExpanderWrapper
from networks import pipeline
from networks.distribution_networks import GaussianMLPIndependentStdModule, GaussianMLPTwoHeadedModule, GaussianMLPGlobalStdModule
from networks.feature_extractors import slot_extractors, slot_poolers
from utils.distributions.tanh import TanhNormal
from utils.weight_initializer.xavier_init import xavier_normal
from RL.policies.policy import Policy
from ReplayBuffers.path_replay_buffer import PathBuffer
from networks.mlp import MLPModule
from skill_discovery.METRA.metra import METRA
from RL.algos.sac import SAC
from RL.algos.ppo import PPO
from networks.parameter import ParameterModule
from gym.vector import AsyncVectorEnv, SyncVectorEnv
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


def fetch_activation(activation_name):
    assert activation_name in ['relu', 'tanh', 'elu'], 'Only relu, elu or tanh are supported as activations'
    if activation_name == 'relu':
        return torch.relu
    elif activation_name == 'tanh':
        return torch.tanh
    elif activation_name == 'elu':
        return torch.nn.ELU


def fetch_dist_type(class_name):
    assert class_name in ['TanhNormal']
    if class_name == 'TanhNormal':
        return TanhNormal
    

def fetch_extractor_and_pooler(extractor_config, pooler_config, channels = None, obs_size = None):
    if extractor_config.name == 'CNN':
        extractor = slot_extractors.CNNExtractor(channels = channels, obs_size = obs_size,
                                                 act = fetch_activation(extractor_config.act),
                                                 norm = extractor_config.norm, cnn_depth = extractor_config.cnn_depth,
                                                 cnn_kernels = extractor_config.cnn_kernels,
                                                 spectral_normalization = extractor_config.spectral_normalisation)
    elif extractor_config.name == 'dummy':
        extractor = slot_extractors.DummySlotExtractor(obs_size)
    else:
        assert False, 'Unknown pooler'

    downstream_obs_size = extractor.outp_dim

    if pooler_config.name == 'transformer':
        pooler = slot_poolers.TransformerSlotAggregator(obs_dim = downstream_obs_size, 
                                                        nhead = pooler_config.nhead, 
                                                        dim_feedforward = pooler_config.dim_feedforward,
                                                        num_layers = pooler_config.num_layers)
    elif pooler_config.name == 'fetcher':
        pooler = slot_poolers.Fetcher(downstream_obs_size)
    else:
        assert False, 'Unknown pooler'
    return extractor, pooler


def build_policy_net(env, policy_net_config):
    obs_dim, action_dim = env.spec.observation_space, env.spec.action_space.flat_dim
    if len(obs_dim.shape) == 3:
        channels, obs_size = obs_dim.shape[2], obs_dim.shape[0:2]
    else:
        obs_dim = obs_dim.flat_dim
        channels, obs_size = None, obs_dim

    extractor, pooler = fetch_extractor_and_pooler(policy_net_config.slot_extractor, policy_net_config.slot_pooler,
                                                   channels = channels, obs_size = obs_size)
    module_obs_dim = pooler.outp_dim + policy_net_config.dim_option
    if policy_net_config.distribution.name == 'TwoHeaded':
        policy_module = GaussianMLPTwoHeadedModule(input_dim = module_obs_dim, output_dim = action_dim, 
                                               hidden_sizes = policy_net_config.hidden_sizes, 
                                               layer_normalization = policy_net_config.layer_normalization,
                                               hidden_nonlinearity = fetch_activation(policy_net_config.nonlinearity), 
                                               max_std = np.exp(policy_net_config.distribution.max_logstd),
                                               init_std = np.exp(policy_net_config.distribution.starting_logstd),
                                               normal_distribution_cls = fetch_dist_type(policy_net_config.distribution.type),
                                               output_w_init = functools.partial(xavier_normal, gain=1.))
    elif policy_net_config.distribution.name == 'GlobalStd':
        policy_module = GaussianMLPGlobalStdModule(input_dim = module_obs_dim, output_dim = action_dim, 
                                               hidden_sizes = policy_net_config.hidden_sizes, 
                                               layer_normalization = policy_net_config.layer_normalization,
                                               hidden_nonlinearity = fetch_activation(policy_net_config.nonlinearity), 
                                               max_std = np.exp(policy_net_config.distribution.max_logstd),
                                               init_std = np.exp(policy_net_config.distribution.starting_logstd),
                                               normal_distribution_cls = fetch_dist_type(policy_net_config.distribution.type),
                                               output_w_init = functools.partial(xavier_normal, gain=1.))
        
    policy_module = pipeline.SkillObjectPipeline('obs', 'options', 'obj_idxs', 
                                                 slot_extractor = extractor, slot_pooler = pooler, 
                                                 downstream_model = policy_module, downstream_input_keys = ['obs', 'options'])
    return Policy(name = policy_net_config.name, module = policy_module)


def build_trajectory_encoder(env, trajectory_net_config):
    obs_dim = env.spec.observation_space
    if len(obs_dim.shape) == 3:
        channels, obs_size = obs_dim.shape[2], obs_dim.shape[0:2]
    else:
        obs_dim = obs_dim.flat_dim
        channels, obs_size = None, obs_dim

    extractor, pooler = fetch_extractor_and_pooler(trajectory_net_config.slot_extractor, trajectory_net_config.slot_pooler,
                                                   channels = channels, obs_size = obs_size)
    module_obs_dim = pooler.outp_dim
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
    
    traj_encoder = pipeline.SkillObjectPipeline('obs', 'options', 'obj_idxs', 
                                                 slot_extractor = extractor, slot_pooler = pooler, 
                                                 downstream_model = traj_encoder, downstream_input_keys = ['obs'])
    return traj_encoder


def build_q_net(env, q_net_config):
    obs_dim, action_dim = env.spec.observation_space, env.spec.action_space.flat_dim
    if len(obs_dim.shape) == 3:
        channels, obs_size = obs_dim.shape[2], obs_dim.shape[0:2]
    else:
        obs_dim = obs_dim.flat_dim
        channels, obs_size = None, obs_dim

    extractor, pooler = fetch_extractor_and_pooler(q_net_config.slot_extractor, q_net_config.slot_pooler,
                                                   channels = channels, obs_size = obs_size)
    module_obs_dim = pooler.outp_dim + q_net_config.dim_option + action_dim

    q = MLPModule(input_dim = module_obs_dim, output_dim = 1,
                  hidden_sizes = q_net_config.hidden_sizes,
                  hidden_nonlinearity = fetch_activation(q_net_config.nonlinearity),
                  hidden_w_init = torch.nn.init.xavier_normal_,
                  hidden_b_init = torch.nn.init.zeros_,
                  output_nonlinearity = None,
                  output_w_init = torch.nn.init.xavier_normal_,
                  output_b_init = torch.nn.init.zeros_,
                  layer_normalization = q_net_config.layer_normalization)
    
    q = pipeline.SkillObjectPipeline('obs', 'options', 'obj_idxs', 
                                     slot_extractor = extractor, slot_pooler = pooler, 
                                     downstream_model = q, downstream_input_keys = ['obs', 'options', 'actions'])
    return q

def build_v_net(env, v_net_config):
    obs_dim = env.spec.observation_space
    if len(obs_dim.shape) == 3:
        channels, obs_size = obs_dim.shape[2], obs_dim.shape[0:2]
    else:
        obs_dim = obs_dim.flat_dim
        channels, obs_size = None, obs_dim

    extractor, pooler = fetch_extractor_and_pooler(v_net_config.slot_extractor, v_net_config.slot_pooler,
                                                   channels = channels, obs_size = obs_size)
    module_obs_dim = pooler.outp_dim + v_net_config.dim_option

    v = MLPModule(input_dim = module_obs_dim, output_dim = 1,
                  hidden_sizes = v_net_config.hidden_sizes,
                  hidden_nonlinearity = fetch_activation(v_net_config.nonlinearity),
                  hidden_w_init = torch.nn.init.xavier_normal_,
                  hidden_b_init = torch.nn.init.zeros_,
                  output_nonlinearity = None,
                  output_w_init = torch.nn.init.xavier_normal_,
                  output_b_init = torch.nn.init.zeros_,
                  layer_normalization = v_net_config.layer_normalization)
    
    v = pipeline.SkillObjectPipeline('obs', 'options', 'obj_idxs', 
                                     slot_extractor = extractor, slot_pooler = pooler, 
                                     downstream_model = v, downstream_input_keys = ['obs', 'options'])
    return v
    

def build_dist_predictor(env, dual_dist_name, dual_dist_config):
    assert dual_dist_name in ['l2', 'one', 's2_from_s'], 's2_from_s and other distances are not supported. Yet...'
    
    if dual_dist_name in ['l2', 'one']:
        return None
    
    elif dual_dist_name == 's2_from_s':
        obs_dim = env.spec.observation_space
        dist_predictor = GaussianMLPIndependentStdModule(hidden_sizes = dual_dist_config.hidden_sizes, 
                                        std_hidden_sizes = dual_dist_config.hidden_sizes,
                                        hidden_nonlinearity = fetch_activation(dual_dist_config.nonlinearity),
                                        std_hidden_nonlinearity = fetch_activation(dual_dist_config.nonlinearity),
                                        std_hidden_w_init = torch.nn.init.xavier_uniform_, 
                                        std_output_w_init = torch.nn.init.xavier_uniform_,
                                        hidden_w_init = torch.nn.init.xavier_uniform_, 
                                        output_w_init = torch.nn.init.xavier_uniform_,
                                        init_std = dual_dist_config.init_std,
                                        min_std = dual_dist_config.min_std,
                                        max_std = dual_dist_config.max_std,
                                        std_parameterization = dual_dist_config.std_parameterization,
                                        bias = dual_dist_config.bias,
                                        input_dim = obs_dim, 
                                        output_dim = obs_dim,
                                        spectral_normalization = dual_dist_config.spectral_normalization)
        return dist_predictor


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
                                        env_kwargs = config.env.env_kwargs,
                                        max_path_length = config.env.max_path_length,
                                        frame_stack = config.env.frame_stack, 
                                        normalizer_type = config.env.normalizer_type)
    env = make_seeded_env(seed = config.globals.seed)

    option_policy = build_policy_net(env, policy_net_config = config.rl_algo.policy)

    traj_encoder = build_trajectory_encoder(env, trajectory_net_config = config.skill.trajectory_encoder)
    METRA_optimizer_keys = ['traj_encoder', 'dual_lam']
    dist_predictor = build_dist_predictor(env, config.skill.dual_dist_name, dual_dist_config = config.skill.dual_dist)
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

    if dist_predictor is not None:
        optimizers['dist_predictor'] = torch.optim.Adam([
            {'params': dist_predictor.parameters(), 'lr': config.skill.dual_dist.lr},
        ])
        METRA_optimizer_keys.append('dist_predictor')
    
    rl_algo = None

    pixel_keys = ['obs', 'next_obs'] if env.is_pixel else []
    if config.rl_algo.name == 'SAC':
        qf1, qf2 = build_q_net(env, q_net_config = config.rl_algo.critics),\
            build_q_net(env, q_net_config = config.rl_algo.critics)
    
        log_alpha = ParameterModule(torch.Tensor([np.log(config.rl_algo.alpha.value)]))
        optimizers.update({
            'qf': torch.optim.Adam([
                {'params': list(qf1.parameters()) + list(qf2.parameters()), 'lr': config.rl_algo.critics.lr},
            ]),
            'log_alpha': torch.optim.Adam([
                {'params': log_alpha.parameters(), 'lr': config.rl_algo.alpha.lr},
            ])
        })

        rl_algo = SAC(qf1 = qf1, qf2 = qf2, log_alpha = log_alpha, tau = config.rl_algo.tau, 
                      scale_reward = config.rl_algo.scale_reward, target_coef = config.rl_algo.target_coef,
                      option_policy = option_policy, device = config.globals.device,
                      optimizers = {key: optimizers[key] for key in ['option_policy', 'qf', 'log_alpha']},
                      discount = config.rl_algo.discount, env_spec = env)
        
        replay_buffer = PathBuffer(capacity_in_transitions = int(config.replay_buffer.common.max_transitions), 
                                   batch_size = config.replay_buffer.common.batch_size, pixel_keys = pixel_keys, 
                                   discount = config.replay_buffer.common.discount, gae_lambda = None,
                                   on_policy = rl_algo.on_policy)
        rollout_buffer = None
    
    elif config.rl_algo.name == 'PPO':
        vf = build_v_net(env, v_net_config = config.rl_algo.value)
        optimizers.update({
            'vf': torch.optim.Adam([
                {'params': list(vf.parameters()), 'lr': config.rl_algo.value.lr}
            ])
        })
        rl_algo = PPO(vf, clip_coef = config.rl_algo.clip_coef, clip_vloss = config.rl_algo.clip_vloss, 
                      ent_coef = config.rl_algo.ent_coef, vf_coef = config.rl_algo.vf_coef, 
                      normalize_advantage = config.rl_algo.norm_adv, max_grad_norm = config.rl_algo.max_grad_norm, 
                      target_kl = config.rl_algo.target_kl, option_policy = option_policy, 
                      optimizers = {key: optimizers[key] for key in ['option_policy', 'vf']},
                      device = config.globals.device)
        
        replay_buffer = PathBuffer(capacity_in_transitions = int(config.replay_buffer.skill.max_transitions), 
                                   batch_size = config.replay_buffer.skill.batch_size, pixel_keys = pixel_keys, 
                                   discount = config.replay_buffer.discount, gae_lambda = config.replay_buffer.gae_lambda,
                                   on_policy = rl_algo.on_policy)
        
        rollout_buffer = PathBuffer(capacity_in_transitions = int(config.replay_buffer.policy.max_transitions), 
                                    batch_size = config.replay_buffer.policy.batch_size, pixel_keys = pixel_keys, 
                                    discount = config.replay_buffer.discount, gae_lambda = config.replay_buffer.gae_lambda,
                                    on_policy = rl_algo.on_policy)
        

    metra = METRA(traj_encoder = traj_encoder, dual_lam = dual_lam, dist_predictor = dist_predictor,
                  optimizers = {key: optimizers[key] for key in METRA_optimizer_keys}, 
                  dim_option = config.skill.dim_option, discrete = config.skill.discrete, 
                  unit_length = config.skill.unit_length, device = config.globals.device, 
                  dual_reg = config.skill.dual_reg, dual_slack = config.skill.dual_slack, 
                  dual_dist = config.skill.dual_dist_name)
        
    env.close()
    
    train_cycle(config.trainer_args, agent = rl_algo, skill_model = metra, replay_buffer = replay_buffer,
                rollout_buffer = rollout_buffer, make_env_fn = make_seeded_env, 
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


def collect_trajectories(env, agent, trajectories_length, n_objects = None, trajectories_qty = None, 
                         skill_model = None, options_and_obj_idxs = None, mode = 'train'):
    if options_and_obj_idxs is None:
        options_and_obj_idxs = skill_model._get_train_trajectories_kwargs(trajectories_qty, n_objects)
    else:
        assert (trajectories_qty is None) and (skill_model is None), """ It is expected, that when using options, 
        there are one trajectory per each option, and there is no need to sample new options """
        trajectories_qty = len(options_and_obj_idxs)
    
    env_qty = len(env.env_fns)
    if isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
        pseudoepisodes = math.ceil(trajectories_qty / len(env.env_fns))
    else:
        pseudoepisodes = trajectories_qty
    
    generated_trajectories = [{} for i in range(trajectories_qty)]
    for pseudoepisode in range(pseudoepisodes):
        cur_options = np.array([options_and_obj_idxs[pseudoepisode * env_qty + i]['options'] for i in range(env_qty)])
        cur_obj_idx = np.array([options_and_obj_idxs[pseudoepisode * env_qty + i]['obj_idxs'] for i in range(env_qty)])
        prev_obs = env.reset()
        prev_obs = np.transpose(prev_obs, [0, 3, 1, 2]) if len(prev_obs.shape) == 4 else prev_obs
        prev_dones = np.full((env_qty,), fill_value=False)
        for i in range(trajectories_length):
            obs_and_option = {'obs': prev_obs, 'options': cur_options, 'obj_idxs': cur_obj_idx}
            action, action_info = agent.policy['option_policy'].get_actions(obs_and_option)
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
                    agent_info.update({'options': cur_options[j], 'obj_idxs': cur_obj_idx[j]})
                    if mode == 'train':
                        agent_info.update({'log_probs': action_info['log_prob'][j]})
                    update_traj_with_info(generated_trajectories[j + pseudoepisode * env_qty], agent_info, 'agent_infos')
            prev_obs = next_obs
            prev_dones = np.logical_or(prev_dones, dones)
    
    generated_trajectories = generated_trajectories[:trajectories_qty]
    if agent.on_policy and mode == 'train':
        prev_obs = torch.tensor([generated_trajectories[i]['observations'] for i in range(len(generated_trajectories))])
        next_obs = torch.tensor([generated_trajectories[i]['next_observations'] for i in range(len(generated_trajectories))])
        options = torch.tensor([generated_trajectories[i]['agent_infos']['options'] for i in range(len(generated_trajectories))])
        obj_idxs = torch.tensor([generated_trajectories[i]['agent_infos']['obj_idxs'] for i in range(len(generated_trajectories))])
        prev_obs = {'obs': prev_obs.reshape((-1,) + tuple(prev_obs.shape[2:])).to(agent.device), 
                    'options': options.reshape((-1,) + tuple(options.shape[2:])).to(agent.device), 
                    'obj_idxs': obj_idxs.reshape((-1,) + tuple(obj_idxs.shape[2:])).to(agent.device)}
        next_obs = {'obs': next_obs.reshape((-1,) + tuple(next_obs.shape[2:])).to(agent.device), 
                    'options': options.reshape((-1,) + tuple(options.shape[2:])).to(agent.device), 
                    'obj_idxs': obj_idxs.reshape((-1,) + tuple(obj_idxs.shape[2:])).to(agent.device)}
        values = agent.critic(prev_obs).detach().cpu().numpy().astype(np.float32).reshape((trajectories_qty, -1))
        next_values = agent.critic(next_obs).detach().cpu().numpy().astype(np.float32).reshape((trajectories_qty, -1))
        agent_info.update({'values': agent.critic(prev_obs).detach().cpu().numpy(),
                           'next_values': agent.critic(next_obs).detach().cpu().numpy()})
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

def train_cycle(trainer_config, agent, skill_model, replay_buffer, rollout_buffer, make_env_fn, seed, 
                comet_logger):
    n_objects = make_env_fn(seed = 0).n_obj
    env = SyncVectorEnv([lambda: make_env_fn(seed = (seed + i)) for i in range(trainer_config.n_parallel)]) #, context='spawn')
    eval_env = SyncVectorEnv([lambda: make_env_fn(seed = (seed + i)) for i in range(trainer_config.n_parallel)]) #, context='spawn')
    
    prev_cur_step, cur_step = 0, 0
    for i in range(trainer_config.n_epochs):
        agent.eval()
        trajs = collect_trajectories(env = env, agent = agent, skill_model = skill_model, n_objects = n_objects,
                            trajectories_qty = trainer_config.traj_batch_size, 
                            trajectories_length = trainer_config.max_path_length)
        for traj in trajs:
            cur_step += len(traj['observations'])
        replay_buffer.update_replay_buffer(trajs)
        if (rollout_buffer is None) and (replay_buffer.n_transitions_stored < trainer_config.transitions_before_training):
            continue

        agent.train()
        skill_optim_steps = trainer_config.skill_optimization_epochs if agent.on_policy \
            else trainer_config.trans_optimization_epochs
        
        for _ in range(skill_optim_steps):
            batch = replay_buffer.sample_transitions()
            batch = prepare_batch(batch)
            logs, modified_batch = skill_model.train_components(batch)
            if not agent.on_policy:
                logs.update(agent.optimize_op(modified_batch))
        
        if agent.on_policy:
            paths = replay_buffer.pop_recent_paths()
            for path in paths:
                with torch.no_grad():
                    torch_path = {'obs': torch.from_numpy(path['observations']).to(agent.device), 
                                  'next_obs': torch.from_numpy(path['next_observations']).to(agent.device), 
                                  'obj_idxs': torch.from_numpy(path['agent_infos']['obj_idxs']).to(agent.device), 
                                  'options': torch.from_numpy(path['agent_infos']['options']).to(agent.device)}
                    skill_model._update_rewards(torch_path)
                    path['rewards'] = torch_path['rewards'].detach().cpu().numpy()
                
            rollout_buffer.update_replay_buffer(paths)
            for i in range(trainer_config.skill_optimization_epochs * trainer_config.policy_optimization_mult):
                batch = rollout_buffer.sample_transitions()
                batch = prepare_batch(batch)
                ppo_log = agent.optimize_op(batch)
                if ppo_log is None:
                    break
                logs.update(ppo_log)
                logs['last_iteration'] = i
            rollout_buffer.clear()
        
        if (prev_cur_step // trainer_config.log_frequency) < (cur_step // trainer_config.log_frequency):
            comet_logger.log_metrics(logs, step = cur_step)
        
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
        random_options = np.random.randn(num_random_trajectories, skill_model.dim_option).astype(np.float32)
        if skill_model.unit_length:
            random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
        random_option_colors = get_option_colors(random_options * 4)
    random_options = [{'options': opt} for opt in random_options]
    return random_options, random_option_colors


def sample_video_options(skill_model):
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
            for angle in [0, 5, 6, 7]:
                video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
            video_options = np.array(video_options)
        else:
            video_options = np.random.randn(8, skill_model.dim_option)
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
            eval_options.append({'options': elem['options'], 'obj_idxs': obj_i})
        random_trajectories.append(collect_trajectories(env = eval_env, agent = agent, trajectories_length = 200, 
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
        last_obs = {'obs': last_obs, 'obj_idxs': last_obj_idx}
        
        option_dists = skill_model.traj_encoder(last_obs)
        option_means = option_dists.mean.detach().cpu().numpy()
        option_stddevs = torch.ones_like(option_dists.stddev.detach().cpu()).numpy()
        option_samples = option_dists.mean.detach().cpu().numpy()

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
    agent.option_policy._force_use_mode_actions = True
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
    agent.option_policy._force_use_mode_actions = False
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
