from comet_ml import Experiment

import torch
import hydra
import omegaconf
import gym
import random, time

import numpy as np

from oc import ocrs
from oc.optimizer.optimizer import OCOptimizer
from RL.policy import Policy
from RL.rollout_buffer import OCRolloutBuffer
from utils.train_tools import infer_obs_action_shape, make_env

@hydra.main(config_path="configs/", config_name="train_rl")
def main(config):
    experiment = Experiment(
        api_key = 'bbCMVUhDwSJsEqwcmhZ2MXdfE',
        project_name = 'test_PPO_cleanRL',
        workspace = 'denmanorwat'
        )
    
    if config.num_envs == 1:
        envs = gym.vector.SyncVectorEnv(
        [lambda: make_env(config.env, gamma = config.sb3.gamma, 
                          ocr_min_val = config.ocr.image_limits[0], ocr_max_val = config.ocr.image_limits[1], 
                          seed = config.seed)]
    )
    else:
        envs = gym.vector.AsyncVectorEnv(
            [lambda rank = i: make_env(config.env, gamma = config.sb3.gamma, 
                                       ocr_min_val = config.ocr.image_limits[0], ocr_max_val = config.ocr.image_limits[1], 
                                       seed = config.seed, rank = rank) for i in range(config.num_envs)],
                                       context = 'fork')

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda")
    
    obs_shape, is_discrete, agent_action_data, action_shape = infer_obs_action_shape(envs)
    # assert len(obs_shape) in [1, 3], 'It is expected that observations are either images (3d) or vectors (1d)'
    rollout_buffer = OCRolloutBuffer(gamma = config.sb3.gamma, gae_lambda = config.sb3.gae_lambda, device = device, seed = config.seed,
                                     num_parallel_envs = config.num_envs, memory_size = 50_000)
    rollout_buffer.initialize_target_shapes(obs_shape = obs_shape, action_shape = action_shape)

    agent = Policy(observation_size = obs_shape[-1], action_size = agent_action_data, is_action_discrete = is_discrete, 
                   actor_mlp = [64, 64], actor_act = 'Tanh', critic_mlp = [64, 64], critic_act = 'Tanh',
                   pooler_config = config.pooling).to(device)
    oc_model = getattr(ocrs, config.ocr.name)(config.ocr, config.env)
    oc_model = oc_model.to('cuda')
    ocr_optimizer, policy_optimizer = omegaconf.OmegaConf.to_container(config.ocr.optimizer), \
        omegaconf.OmegaConf.to_container(config.sb3.optimizer)
    all_optimizer_config = {**ocr_optimizer, **policy_optimizer}

    optimizer = OCOptimizer(all_optimizer_config, oc_model = oc_model, policy = agent)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, next_done = torch.Tensor(envs.reset()).to(device), torch.zeros(config.num_envs).to(device)

    for iteration in range(1, int(config.max_steps + 1) // config.sb3.n_steps):
        # Annealing the rate if instructed to do so.
        for step in range(0, config.sb3.n_steps, config.num_envs):
            global_step += config.num_envs
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
            next_value = agent.get_value(next_obs)
        rollout_buffer.finalize_tensors_calculate_and_store_GAE(last_done = next_done, 
                                                                last_value = next_value)
        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(config.sb3.max_epochs):
            for batch in rollout_buffer.convert_transitions_to_rollout(batch_size = config.sb3.batch_size):
                _, newlogprob, entropy = agent.get_action_logprob_entropy(batch['obs'], batch['action'])
                newvalue = agent.get_value(batch['obs'])
                logratio = newlogprob - batch['logprob']
                ratio = logratio.exp()
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.sb3.clip_range).float().mean().item()]
                normalized_advantages = (batch['advantage'] - batch['advantage'].mean()) / (batch['advantage'].std() + 1e-8)
                # Policy loss
                pg_loss1 = -normalized_advantages * ratio
                pg_loss2 = -normalized_advantages * torch.clamp(ratio, 1 - config.sb3.clip_range, 1 + config.sb3.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((newvalue - batch['return']) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - config.sb3.ent_coef * entropy_loss + v_loss * config.sb3.vf_coef
                optimizer.optimizer_zero_grad()
                loss.backward()
                optimizer.optimizer_step('rl')

        y_true, y_pred = rollout_buffer.get_return_value()
        y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        rollout_buffer.reset_trajectories()
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        experiment.log_metric(name = "losses/value_loss", value = v_loss.item(), step = global_step)
        experiment.log_metric(name = "losses/policy_loss", value = pg_loss.item(), step = global_step)
        experiment.log_metric(name = "losses/entropy", value = entropy_loss.item(), step = global_step)
        experiment.log_metric(name = "losses/old_approx_kl", value = old_approx_kl.item(), step = global_step)
        experiment.log_metric(name = "losses/approx_kl", value = approx_kl.item(), step = global_step)
        experiment.log_metric(name = "losses/clipfrac", value = np.mean(clipfracs), step = global_step)
        experiment.log_metric(name = "losses/explained_variance", value = explained_var, step = global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        experiment.log_metric(name = "charts/SPS", value = int(global_step / (time.time() - start_time)), step = global_step)

if __name__ == "__main__":
    main()