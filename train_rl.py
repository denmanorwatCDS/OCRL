import torch
import gym
import random, time

import numpy as np
from torch import nn, optim
from comet_ml import Experiment

from RL.policy import Policy
from RL.rollout_buffer import OCRolloutBuffer

env_id =  "CartPole-v1" # "HalfCheetah-v3"
capture_video = False
run_name = 'CheetahTest'
gamma = 0.99
gae_lambda = 0.95
num_envs = 1
num_steps = 2048
learning_rate = 3e-4
seed = 1
total_timesteps = int(1e6)
anneal_lr = True
update_epochs = 10
num_minibatches = 32
clip_coef = 0.2
norm_adv = True
clip_vloss = True
ent_coef = 0.
vf_coef = 0.5
max_grad_norm = 0.5
target_kl = None

batch_size = int(num_envs * num_steps)
num_iterations = total_timesteps // batch_size
minibatch_size = int(batch_size // num_minibatches)

experiment = Experiment(
    api_key = 'bbCMVUhDwSJsEqwcmhZ2MXdfE',
    project_name = 'test_PPO_cleanRL',
    workspace = 'denmanorwat'
    )

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        if 'Cheetah' in env_id:
            env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        else:
            env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def infer_action(envs):
    if isinstance(envs.action_space, gym.spaces.tuple.Tuple):
        if np.all([isinstance(envs.action_space[i], gym.spaces.discrete.Discrete) 
                   for i in range(len(envs.action_space))]):
            example_action = torch.Tensor(envs.action_space.sample())[0].unsqueeze(-1)
            return example_action
        elif np.all([isinstance(envs.action_space[i], gym.spaces.Box) 
                     for i in range(len(envs.action_space))]):
            return torch.Tensor(envs.action_space.sample()[0])
    
    elif isinstance(envs.action_space, gym.spaces.Box):
        return torch.Tensor(envs.action_space.sample()[0])

    else:
        assert False, 'What da faq?'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda")

# env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(env_id, i, capture_video, run_name, gamma) for i in range(num_envs)]
)

if isinstance(envs.single_action_space, gym.spaces.Box):
    is_action_discrete = False
    action_space_size = envs.action_space[0].shape[-1]
else:
    is_action_discrete = True
    action_space_size = envs.action_space[0].n

agent = Policy(envs.observation_space.shape[-1], action_space_size, is_action_discrete = is_action_discrete, 
               actor_mlp = [64, 64], actor_act = 'Tanh', critic_mlp = [64, 64], critic_act = 'Tanh',
               pooler_config = {'name': 'IdentityPooler'}).to(device)
optimizer = optim.Adam(agent.parameters(), lr = learning_rate, eps = 1e-5)
rollout_buffer = OCRolloutBuffer(gamma = gamma, gae_lambda = gae_lambda, device = device, seed = seed,
                                 num_parallel_envs = num_envs, memory_size = 50_000)

# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
next_obs = envs.reset()
next_obs = torch.Tensor(next_obs).to(device)
next_done = torch.zeros(num_envs).to(device)
rollout_buffer.initialize_target_shapes(obs = next_obs, action = agent.get_action_logprob_entropy(next_obs)[0])
for iteration in range(1, num_iterations + 1):
    # Annealing the rate if instructed to do so.
    if anneal_lr:
        frac = 1.0 - (iteration - 1.0) / num_iterations
        lrnow = frac * learning_rate
        optimizer.param_groups[0]["lr"] = lrnow
    for step in range(0, num_steps):
        global_step += num_envs
        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, entropy = agent.get_action_logprob_entropy(next_obs)
            value = agent.get_value(next_obs)
        tran = {'obs': next_obs, 'done': next_done, 'action': action, 'logprob': logprob, 'value': value}
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, next_done, infos = envs.step(action.cpu().numpy())
        tran['reward'] = torch.Tensor(reward).to(device)
        rollout_buffer.save_transition(tran)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
        if torch.any(next_done):
            for i, info in enumerate(infos):
                if next_done[i] and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    experiment.log_metric("charts/episodic_return", info["episode"]["r"], global_step)
                    experiment.log_metric("charts/episodic_length", info["episode"]["l"], global_step)
    # bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
    rollout_buffer.finalize_tensors_calculate_and_store_GAE(last_done = next_done, 
                                                            last_value = next_value)
    # flatten the batch
    b_obs = rollout_buffer._trajectories['obs'].reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = rollout_buffer._trajectories['logprob'].reshape(-1)
    b_actions = rollout_buffer._trajectories['action'].reshape((-1,) + envs.single_action_space.shape)
    b_advantages = rollout_buffer._trajectories['advantage'].reshape(-1)
    b_returns = rollout_buffer._trajectories['return'].reshape(-1)
    b_values = rollout_buffer._trajectories['value'].reshape(-1)
    # Optimizing the policy and value network
    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            _, newlogprob, entropy = agent.get_action_logprob_entropy(b_obs[mb_inds], b_actions[mb_inds])
            newvalue = agent.get_value(b_obs[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]
            mb_advantages = b_advantages[mb_inds]
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            # Value loss
            newvalue = newvalue.view(-1)
            if clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
        if target_kl is not None and approx_kl > target_kl:
            break

    rollout_buffer.reset_trajectories()
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    # TRY NOT TO MODIFY: record rewards for plotting purposes
    experiment.log_metric(name = "charts/learning_rate", value = optimizer.param_groups[0]["lr"], step = global_step)
    experiment.log_metric(name = "losses/value_loss", value = v_loss.item(), step = global_step)
    experiment.log_metric(name = "losses/policy_loss", value = pg_loss.item(), step = global_step)
    experiment.log_metric(name = "losses/entropy", value = entropy_loss.item(), step = global_step)
    experiment.log_metric(name = "losses/old_approx_kl", value = old_approx_kl.item(), step = global_step)
    experiment.log_metric(name = "losses/approx_kl", value = approx_kl.item(), step = global_step)
    experiment.log_metric(name = "losses/clipfrac", value = np.mean(clipfracs), step = global_step)
    experiment.log_metric(name = "losses/explained_variance", value = explained_var, step = global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    experiment.log_metric(name = "charts/SPS", value = int(global_step / (time.time() - start_time)), step = global_step)