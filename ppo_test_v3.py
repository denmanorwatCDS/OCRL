# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
import argparse
import omegaconf
import pathlib
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ppo_test_wrappers.wrappers import NormalizeObservation, NormalizeReward
import os
from utils.distributions.tanh import TanhNormal, PPOTanhNormal
from networks.feature_extractors import slot_extractors, slot_poolers
import functools
from torch.nn.init import orthogonal_
from utils.weight_initializer.xavier_init import xavier_normal
from networks.distribution_networks import GaussianMLPIndependendInputStdModule, \
    GaussianMLPTwoHeadedModule, GaussianMLPGlobalStdModule
import math
from networks import pipeline
from RL.policies.policy import Policy
from networks.mlp import MLPModule


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v2"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def fetch_activation(activation_name):
    assert activation_name in ['relu', 'tanh', 'elu'], 'Only relu, elu or tanh are supported as activations'
    if activation_name == 'relu':
        return torch.relu
    elif activation_name == 'tanh':
        return torch.tanh
    elif activation_name == 'elu':
        return torch.nn.ELU


def fetch_dist_type(class_name):
    assert class_name in ['TanhNormal', 'PPOTanhNormal']
    if class_name == 'TanhNormal':
        return TanhNormal
    if class_name == 'PPOTanhNormal':
        return PPOTanhNormal
    

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


def fetch_init(name, gain = None):
    if name == 'xavier':
        if gain is None:
            gain = 1.
        return functools.partial(xavier_normal, gain = gain)
    elif name == 'orthogonal':
        if gain is None:
            gain = math.sqrt(2)
        return functools.partial(orthogonal_, gain = gain)


def build_policy_net(env, policy_net_config):
    obs_dim, action_dim = env.observation_space, env.action_space
    if len(obs_dim.shape) == 3:
        channels, obs_size = obs_dim.shape[2], obs_dim.shape[0:2]
    else:
        obs_dim = np.array(obs_dim.shape[1])
        action_dim = np.array(action_dim[0].shape[0])
        channels, obs_size = None, obs_dim

    extractor, pooler = fetch_extractor_and_pooler(policy_net_config.slot_extractor, policy_net_config.slot_pooler,
                                                   channels = channels, obs_size = obs_size)
    module_obs_dim = pooler.outp_dim
    if policy_net_config.distribution.name == 'TwoHeaded':
        policy_module = GaussianMLPTwoHeadedModule(input_dim = module_obs_dim, output_dim = action_dim, 
                                               hidden_sizes = policy_net_config.hidden_sizes, 
                                               layer_normalization = policy_net_config.layer_normalization,
                                               hidden_nonlinearity = fetch_activation(policy_net_config.nonlinearity),
                                               hidden_w_init = fetch_init(name = policy_net_config.hidden_w_init.name),
                                               output_w_init = fetch_init(name = policy_net_config.output_w_init.name, 
                                                                          gain = policy_net_config.output_w_init.std), 
                                               max_std = np.exp(policy_net_config.distribution.max_logstd) \
                                               if policy_net_config.distribution.max_logstd is not None else None,
                                               min_std = np.exp(policy_net_config.distribution.min_logstd) \
                                               if policy_net_config.distribution.min_logstd is not None else None,
                                               init_std = np.exp(policy_net_config.distribution.starting_logstd),
                                               std_parameterization = policy_net_config.distribution.std_parameterization,
                                               normal_distribution_cls = fetch_dist_type(policy_net_config.distribution.type))
        
        
    elif policy_net_config.distribution.name == 'GlobalStd':
        policy_module = GaussianMLPGlobalStdModule(input_dim = module_obs_dim, output_dim = action_dim, 
                                               hidden_sizes = policy_net_config.hidden_sizes, 
                                               layer_normalization = policy_net_config.layer_normalization,
                                               hidden_nonlinearity = fetch_activation(policy_net_config.nonlinearity),
                                               hidden_w_init = fetch_init(name = policy_net_config.hidden_w_init.name),
                                               output_w_init = fetch_init(name = policy_net_config.output_w_init.name, 
                                                                          gain = policy_net_config.output_w_init.std),
                                               max_std = np.exp(policy_net_config.distribution.max_logstd),
                                               init_std = np.exp(policy_net_config.distribution.starting_logstd),
                                               std_parameterization = policy_net_config.distribution.std_parameterization,
                                               normal_distribution_cls = fetch_dist_type(policy_net_config.distribution.type))
        

    elif policy_net_config.distribution.name == 'SkillStd':
        policy_module = GaussianMLPIndependendInputStdModule(input_idx_mean = [0, module_obs_dim], 
                                                             input_idx_std = [pooler.outp_dim, pooler.outp_dim + policy_net_config.dim_option],
                                                             output_dim = action_dim,
                                                             hidden_sizes = policy_net_config.hidden_sizes, 
                                                             layer_normalization = policy_net_config.layer_normalization,
                                                             hidden_nonlinearity = fetch_activation(policy_net_config.nonlinearity),
                                                             hidden_w_init = fetch_init(name = policy_net_config.hidden_w_init.name),
                                                             output_w_init = fetch_init(name = policy_net_config.output_w_init.name, 
                                                                          gain = policy_net_config.output_w_init.std),
                                                             max_std = np.exp(policy_net_config.distribution.max_logstd) \
                                                             if policy_net_config.distribution.max_logstd is not None else None,
                                                             min_std = np.exp(policy_net_config.distribution.min_logstd) \
                                                             if policy_net_config.distribution.min_logstd is not None else None,
                                                             init_std = np.exp(policy_net_config.distribution.starting_logstd),
                                                             std_parameterization = policy_net_config.distribution.std_parameterization,
                                                             normal_distribution_cls = fetch_dist_type(policy_net_config.distribution.type))
        
    policy_module = pipeline.DictPipeline(policy_module)
    return Policy(name = policy_net_config.name, module = policy_module)


def build_v_net(env, v_net_config):
    obs_dim = env.observation_space
    if len(obs_dim.shape) == 3:
        channels, obs_size = obs_dim.shape[2], obs_dim.shape[0:2]
    else:
        obs_dim = np.array(obs_dim.shape[1])
        channels, obs_size = None, obs_dim

    extractor, pooler = fetch_extractor_and_pooler(v_net_config.slot_extractor, v_net_config.slot_pooler,
                                                   channels = channels, obs_size = obs_size)
    module_obs_dim = pooler.outp_dim

    v = MLPModule(input_dim = module_obs_dim, output_dim = 1,
                  hidden_sizes = v_net_config.hidden_sizes,
                  hidden_nonlinearity = fetch_activation(v_net_config.nonlinearity),
                  hidden_w_init = fetch_init(name = v_net_config.hidden_w_init.name),
                  hidden_b_init = torch.nn.init.zeros_,
                  output_nonlinearity = None,
                  output_w_init = fetch_init(name = v_net_config.output_w_init.name,
                                             gain = v_net_config.output_w_init.std),
                  output_b_init = torch.nn.init.zeros_,
                  layer_normalization = v_net_config.layer_normalization)
    
    v = pipeline.DictPipeline(v)
    return v


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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


if __name__ == "__main__":
    config = fetch_config()
    import comet_ml
    exp = comet_ml.start(project_name = 'PPO_cleanrl')

    # TRY NOT TO MODIFY: seeding
    random.seed(config.globals.seed)
    np.random.seed(config.globals.seed)
    torch.manual_seed(config.globals.seed)
    torch.backends.cudnn.deterministic = False

    device = torch.device("cuda")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env("HalfCheetah-v2", i, False, None, config.replay_buffer.discount) for i in range(1)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = build_policy_net(envs, config.rl_algo.policy).to(device)
    critic = build_v_net(envs, config.rl_algo.value).to(device)
    optimizer = optim.Adam([{'params': actor.get_policy_parameters_without_std(), 'lr': config.rl_algo.policy.lr, 'eps': 1e-05},
                            {'params': actor.get_policy_std_parameters(), 'lr': config.rl_algo.policy.std_lr, 'eps': 1e-05},
                            {'params': critic.parameters(), 'lr': config.rl_algo.value.lr, 'eps': 1e-05}])

    # ALGO Logic: Storage setup
    obs = torch.zeros((config.trainer_args.max_path_length, 1) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config.trainer_args.max_path_length, 1) + envs.single_action_space.shape).to(device)
    pre_tanh_values = torch.zeros((config.trainer_args.max_path_length,) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config.trainer_args.max_path_length, 1)).to(device)
    rewards = torch.zeros((config.trainer_args.max_path_length, 1)).to(device)
    dones = torch.zeros((config.trainer_args.max_path_length, 1)).to(device)
    values = torch.zeros((config.trainer_args.max_path_length, 1)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)

    for iteration in range(1, config.trainer_args.n_epochs + 1):
        # TODO No annealing!!!

        for step in range(0, config.trainer_args.max_path_length):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, info = actor.get_actions({'obs': next_obs})
                logprob, pre_tanh_value = info['log_prob'], info['pre_tanh_value']
                value = critic({'obs': next_obs})
                values[step] = value.flatten()
            actions[step] = torch.from_numpy(action)
            logprobs[step] = torch.from_numpy(logprob)
            pre_tanh_values[step] = torch.from_numpy(pre_tanh_value)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, infos = envs.step(action)
            next_done = done
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if done:
                print(f"global_step={global_step}, episodic_return={infos[0]['episode']['r']}")
                exp.log_metrics({"charts/episodic_return": infos[0]["episode"]["r"],
                                 "charts/episodic_length": infos[0]["episode"]["l"]}, step = global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = critic({'obs': next_obs}).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.trainer_args.max_path_length)):
                if t == config.trainer_args.max_path_length - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.replay_buffer.discount * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.replay_buffer.discount * config.replay_buffer.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_pre_tanh_values = pre_tanh_values.reshape((-1,) + envs.single_action_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config.trainer_args.max_path_length)
        clipfracs = []
        for epoch in range(config.trainer_args.policy_optimization_mult):
            np.random.shuffle(b_inds)
            for start in range(0, config.trainer_args.max_path_length, config.replay_buffer.policy.batch_size):
                end = start + config.replay_buffer.policy.batch_size
                mb_inds = b_inds[start:end]

                newlogprob, entropy, _ = actor.get_logprob_and_entropy({'obs': b_obs[mb_inds]}, 
                                                                     b_actions[mb_inds], b_pre_tanh_values[mb_inds])
                newvalue = critic({'obs': b_obs[mb_inds]})
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.rl_algo.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config.rl_algo.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.rl_algo.clip_coef, 1 + config.rl_algo.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.rl_algo.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.rl_algo.clip_coef,
                        config.rl_algo.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.rl_algo.ent_coef * entropy_loss + v_loss * config.rl_algo.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), config.rl_algo.max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), config.rl_algo.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        exp.log_metrics({"charts/learning_rate": optimizer.param_groups[0]["lr"],
                         "losses/value_loss": v_loss.item(),
                         "losses/policy_loss": pg_loss.item(),
                         "losses/entropy": entropy_loss.item(),
                         "losses/old_approx_kl": old_approx_kl.item(),
                         "losses/approx_kl": approx_kl.item(),
                         "losses/clipfrac": np.mean(clipfracs),
                         "losses/explained_variance": explained_var}, step = global_step)

    envs.close()