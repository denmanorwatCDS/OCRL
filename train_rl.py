from comet_ml import Experiment
from copy import deepcopy
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader
import torch.nn.functional as F

import sys
import h5py
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
from utils.train_tools import infer_obs_action_shape, make_env, update_curves_, get_uint_to_float, stop_oc_optimizer_
from data_utils.H5_dataset import H5Dataset
from utils.eval_tools import calculate_explained_variance, evaluate_ocr_model, get_episodic_metrics, log_ppg_results, add_prefix, Metrics

@hydra.main(config_path="configs/", config_name="train_rl")
def main(config):
    PATH_TO_TRAIN_RL_DOT_PY = sys.path[0]
    experiment = Experiment(
        api_key = 'bbCMVUhDwSJsEqwcmhZ2MXdfE',
        project_name = 'Refactored_OCRL',
        workspace = 'denmanorwat'
        )
    dataset_path = '/'.join([config.dataset_root_path, 
                             f"{config.env.obs_size}x{config.env.obs_size}", 
                             config.env.precollected_dataset, 'dataset', 'data.hdf5'])
    uint_to_float = get_uint_to_float(config.ocr.image_limits[0], config.ocr.image_limits[1])
    rollout_dataset = h5py.File(dataset_path, "r")['TrainingSet']
    # TODO pass rollout preprocessor as input in wrappers to unify preprocessing pipeline for images
    # gathered both from environment and via random policy as they are from same distribution
    # TODO add freezed version of RL algorithm;
    # TODO detach target tensors in PPG phase
    rollout_preprocessor = lambda x: uint_to_float(torch.from_numpy(x)).permute(2, 0, 1)
    
    val_dataset = H5Dataset(datafile = dataset_path, uint_to_float = uint_to_float,
                            use_future = False, 
                            future_steps = config.ocr.slotattr.matching_loss.steps_into_future, 
                            augment = None, is_train = False)
    val_dataloader = DataLoader(val_dataset, batch_size = config.ocr.batch_size, shuffle = False)
    if config.num_envs == 1:
        envs = gym.vector.SyncVectorEnv(
        [lambda: make_env(config.env, gamma = config.sb3.gamma, 
                          ocr_min_val = config.ocr.image_limits[0], ocr_max_val = config.ocr.image_limits[1], 
                          seed = config.seed)])
    else:
        # Due to lazy execution, we pass index of environment explicitly as input to lambda function in order to
        # avoid seeding problems (i.e., identical seed in different processes).
        envs = gym.vector.AsyncVectorEnv(
            [lambda rank = i: make_env(config.env, gamma = config.sb3.gamma, 
                                       ocr_min_val = config.ocr.image_limits[0], ocr_max_val = config.ocr.image_limits[1], 
                                       seed = config.seed, rank = rank) for i in range(config.num_envs)],
                                       context = 'fork')

    random.seed(config.seed), np.random.seed(config.seed), torch.manual_seed(config.seed)
    device = torch.device("cuda")
    
    obs_shape, is_discrete, agent_action_data, action_shape = infer_obs_action_shape(envs)
    rollout_buffer = OCRolloutBuffer(obs_shape = obs_shape, action_shape = action_shape,
                                     gamma = config.sb3.gamma, gae_lambda = config.sb3.gae_lambda, device = device, 
                                     seed = config.seed, num_parallel_envs = config.num_envs,
                                     batch_size = config.sb3.batch_size, 
                                     rollout_max_epochs = config.sb3.rollout_epochs, ppg_max_epochs = config.sb3.ppg_epochs,
                                     use_future = config.ocr.slotattr.matching_loss.use, 
                                     steps_into_future = config.ocr.slotattr.matching_loss.steps_into_future, 
                                     memory_size = 50_000, 
                                     random_dataset = rollout_dataset, dataset_preprocessor = rollout_preprocessor)

    agent = Policy(observation_size = obs_shape[-1], action_size = agent_action_data, is_action_discrete = is_discrete, 
                   actor_mlp = [64, 64], actor_act = 'Tanh', critic_mlp = [64, 64], critic_act = 'Tanh',
                   pooler_config = config.pooling, 
                   ocr_rep_dim = config.ocr.slotattr.slot_size, num_slots = config.ocr.slotattr.num_slots).to(device)
    oc_model = getattr(ocrs, config.ocr.name)(config.ocr, obs_size = config.env.obs_size, obs_channels = config.env.obs_channels)
    if config.pretrained_model.use:
        pretrained_path = '/'.join([PATH_TO_TRAIN_RL_DOT_PY, 'models', 
                                    f'{config.env.obs_size}x{config.env.obs_size}', config.pretrained_model.env,
                                    config.ocr.name + '_' + config.pretrained_model.save_name + ';step:' + config.pretrained_model.step])
        oc_model.load_state_dict(torch.load(pretrained_path, weights_only = True))

    oc_model = oc_model.to('cuda')
    ocr_optimizer, policy_optimizer = omegaconf.OmegaConf.to_container(config.ocr.optimizer), \
        omegaconf.OmegaConf.to_container(config.sb3.optimizer)
    if not config.sb3.train_feature_extractor:
        stop_oc_optimizer_(oc_model, ocr_optimizer)
    all_optimizer_config = {**ocr_optimizer, **policy_optimizer}

    optimizer = OCOptimizer(all_optimizer_config, oc_model = oc_model, policy = agent)

    global_step = 0
    start_time = time.time()
    next_obs, next_done = torch.Tensor(envs.reset()).to(device), torch.zeros(config.num_envs).to(device)
    
    metrics = Metrics()
    for iteration in range(1, int(config.max_steps + 1) // config.sb3.n_steps):
        for step in range(0, config.sb3.n_steps, config.num_envs):
            global_step += config.num_envs

            with torch.no_grad():
                slots = oc_model.get_slots(next_obs, training = False)
                action, logprob, entropy = agent.get_action_logprob_entropy(slots)
                value = agent.get_value(slots)
            
            tran = {'obs': next_obs, 'done': next_done, 'action': action, 'logprob': logprob, 'value': value}
            next_obs, reward, next_done, infos = envs.step(action.cpu().numpy())
            tran['reward'] = torch.Tensor(reward).to(device)
            
            rollout_buffer.save_transition(tran)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            metrics.multiple_update(get_episodic_metrics(next_done, infos))
        
        with torch.no_grad():
            next_value = agent.get_value(slots)
        rollout_buffer.finalize_tensors_calculate_and_store_GAE(last_done = next_done, 
                                                                last_value = next_value)
        
        for batch, start_obs, future_obs in rollout_buffer.convert_transitions_to_rollout():
            slots = oc_model.get_slots(batch['obs'], training = False)
            if not config.sb3.train_feature_extractor:
                slots = slots.detach()
            _, newlogprob, entropy = agent.get_action_logprob_entropy(slots, batch['action'])
            newvalue = agent.get_value(slots)
            
            logratio = newlogprob - batch['logprob']
            ratio = logratio.exp()
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs = ((ratio - 1.0).abs() > config.sb3.clip_range).float().mean().item()
            
            normalized_advantages = (batch['advantage'] - batch['advantage'].mean()) / (batch['advantage'].std() + 1e-8)
            pg_loss1 = -normalized_advantages * ratio
            pg_loss2 = -normalized_advantages * torch.clamp(ratio, 1 - config.sb3.clip_range, 1 + config.sb3.clip_range)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = 0.5 * ((newvalue - batch['return']) ** 2).mean()
            entropy_loss = entropy.mean()
            oc_loss, mets = oc_model.get_loss(obs = start_obs, future_obs = future_obs, do_dropout = True)
            mets = add_prefix(mets, 'oc')
            if not config.sb3.train_feature_extractor:
                oc_loss = 0
            loss = pg_loss - config.sb3.ent_coef * entropy_loss + config.sb3.vf_coef * v_loss + oc_loss
            optimizer.optimizer_zero_grad()
            loss.backward()
            metrics.update(optimizer.optimizer_step('rl'))
            metrics.update(mets)
            metrics.update({'ppo/value_loss': v_loss.item(), 'ppo/policy_gradient_loss': pg_loss.item(), 
                            'ppo/entropy_loss': entropy_loss.item(),
                            'ppo/approx_kl': approx_kl.item(), 'ppo/old_approx_kl': old_approx_kl.item(), 
                            'ppo/clip_fraction': clipfracs})

        if oc_model.requires_ppg() and iteration % config.sb3.ppg_freq:
            optimizer.optimizer_zero_grad()
            # TODO check that target models preserve their weights
            target_oc_model, target_agent = deepcopy(oc_model), deepcopy(agent)
            target_oc_model.inference_mode(), target_agent.inference_mode()
            ppg_curves = {}
            logs_before_ppg, imgs_before_ppg = evaluate_ocr_model(oc_model, val_dataloader)
            for start_obs, future_obs in rollout_buffer.get_obs_generator(mode = 'ppg'):
                if not config.sb3.train_feature_extractor:
                    break
                with torch.no_grad():
                    target_slots = target_oc_model.get_slots(start_obs, training = True)
                    target_distribution = target_agent.get_action_distribution(target_slots)
                    target_values = target_agent.get_value(target_slots)
                oc_loss, loss_metrics = oc_model.get_loss(obs = start_obs, future_obs = future_obs, do_dropout = True)
                slots = oc_model.get_slots(start_obs, training = True)
                distribution, values = agent.get_action_distribution(slots), agent.get_value(slots)
                kl_loss = torch.mean(kl_divergence(target_distribution, distribution))
                vf_loss = config.sb3.vf_coef * F.mse_loss(values, target_values)
                loss_metrics.update({'kl_divergence': kl_loss, 'vf_mse': vf_loss})
                update_curves_(curve_dict = ppg_curves, metrics = loss_metrics)
                total_loss = oc_loss + kl_loss + vf_loss

                optimizer.optimizer_zero_grad()
                total_loss.backward()
                metrics.update(optimizer.optimizer_step('oc'))
            if config.sb3.train_feature_extractor:
                optimizer.reset_optimizers()
            logs_after_ppg, imgs_after_ppg = evaluate_ocr_model(oc_model, val_dataloader)
            log_ppg_results(experiment = experiment, step = global_step, 
                            logs_before_ppg = logs_before_ppg, imgs_before_ppg = imgs_before_ppg,
                            logs_after_ppg = logs_after_ppg, imgs_after_ppg = imgs_after_ppg,
                            curves = ppg_curves)

        y_true, y_pred = rollout_buffer.get_return_value()
        explained_variance = calculate_explained_variance(y_true, y_pred)
        rollout_buffer.reset_trajectories()
        metrics.update({'ppo/explained_variance': explained_variance, 
                        'ppo/steps_per_second': int(global_step / (time.time() - start_time))})
        experiment.log_metrics(metrics.convert_to_dict(), step = global_step)
        
if __name__ == "__main__":
    main()