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
from utils.train_tools import infer_obs_action_shape, make_env, update_curves_, get_uint_to_float, get_float_to_uint, stop_oc_optimizer_
from data_utils.H5_dataset import H5Dataset
from utils.eval_tools import calculate_explained_variance, evaluate_ocr_model, evaluate_agent, get_episodic_metrics, log_oc_results, log_ppg_results, Metrics

@hydra.main(config_path="configs/", config_name="train_rl")
def main(config):
    PATH_TO_TRAIN_RL_DOT_PY = sys.path[0]
    experiment = Experiment(
        api_key = 'bbCMVUhDwSJsEqwcmhZ2MXdfE',
        project_name = 'OC_RL',
        workspace = 'denmanorwat'
        )
    frozen_name = 'Unfrozen' if config.sb3.train_feature_extractor else 'Frozen'
    slot_name = ':'.join(['slot', config.ocr.slotattr.preinit_type])
    wd_array = []
    for key in config.ocr.optimizer.keys():
        if '_optimizer' in key:
            wd_array.append(':'.join([key.replace('_optimizer', '') + '_wd', 
                                      str(config.ocr.optimizer[key]['weight_decay'])]))
    dropout_name = ':'.join(['drop_proba', str(config.ocr.feature_dropout.feature_dropout_proba)])
    name = ' '.join([config.env.env, config.ocr.name, frozen_name, slot_name, *wd_array, dropout_name])
    experiment.set_name(name)
    dataset_path = '/'.join([config.dataset_root_path, 
                             f"{config.env.obs_size}x{config.env.obs_size}", 
                             config.env.precollected_dataset, 'dataset', 'data.hdf5'])
    torch_uint_to_float, numpy_uint_to_float = get_uint_to_float(config.ocr.image_limits[0], config.ocr.image_limits[1])
    float_to_uint = get_float_to_uint(config.ocr.image_limits[0], config.ocr.image_limits[1])
    rollout_dataset = h5py.File(dataset_path, "r")['TrainingSet']
    # TODO pass rollout preprocessor as input in wrappers to unify preprocessing pipeline for images
    # gathered both from environment and via random policy as they are from same distribution
    # TODO add freezed version of RL algorithm;
    # TODO detach target tensors in PPG phase
    rollout_preprocessor = lambda x: torch_uint_to_float(torch.from_numpy(x)).permute(2, 0, 1)
    
    val_dataset = H5Dataset(datafile = dataset_path, uint_to_float = torch_uint_to_float,
                            use_future = False, 
                            future_steps = config.ocr.slotattr.matching_loss.steps_into_future, 
                            augment = None, is_train = False)
    val_dataloader = DataLoader(val_dataset, batch_size = config.ocr.batch_size, shuffle = False)
    if config.num_envs == 1:
        envs = gym.vector.SyncVectorEnv(
        [lambda: make_env(config.env, gamma = config.sb3.gamma, 
                          obs_preprocessor = numpy_uint_to_float, seed = config.seed)])
        eval_env_fns = [lambda: make_env(config.env, gamma = config.sb3.gamma, 
                                         obs_preprocessor = numpy_uint_to_float, seed = config.seed + 1)]
    else:
        # Due to lazy execution, we pass index of environment explicitly as input to lambda function in order to
        # avoid seeding problems (i.e., identical seed in different processes).
        envs = gym.vector.AsyncVectorEnv(
            [lambda rank = i: make_env(config.env, gamma = config.sb3.gamma, 
                                       obs_preprocessor = numpy_uint_to_float, 
                                       seed = config.seed, rank = rank) for i in range(config.num_envs)],
                                       context = 'fork')
        eval_env_fns = [lambda rank = i: make_env(config.env, gamma = config.sb3.gamma, 
                        obs_preprocessor = numpy_uint_to_float, seed = config.seed, rank = rank) \
                        for i in range(config.num_envs, 2 * config.num_envs)]

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

    oc_model = getattr(ocrs, config.ocr.name)(config.ocr, obs_size = config.env.obs_size, obs_channels = config.env.obs_channels)
    agent = Policy(observation_size = obs_shape[-1], action_size = agent_action_data, is_action_discrete = is_discrete, 
                   backbone_mlp = config.sb3_acnet.backbone_net.dims, backbone_act = config.sb3_acnet.backbone_net.act, 
                   actor_mlp = config.sb3_acnet.policy_net.dims, actor_act = config.sb3_acnet.policy_net.act, 
                   critic_mlp = config.sb3_acnet.value_net.dims, critic_act = config.sb3_acnet.value_net.act,
                   pooler_config = config.pooling, 
                   ocr_rep_dim = config.ocr.slotattr.slot_size, num_slots = config.ocr.slotattr.num_slots).to(device)
    if config.pretrained_model.use:
        pretrained_path = '/'.join([PATH_TO_TRAIN_RL_DOT_PY, 'models', 
                                    f'{config.env.obs_size}x{config.env.obs_size}', config.pretrained_model.env,
                                    config.ocr.name + '_' + config.pretrained_model.save_name + ';step:' + config.pretrained_model.step])
        if not ('orig' in config.pretrained_model.save_name):
            oc_model.load_state_dict(torch.load(pretrained_path, weights_only = True))
        else:
            oc_model.load_jaesik_model(torch.load(pretrained_path, map_location = 'cuda:0'))

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
    oc_model.inference_mode()
    oc_logs, oc_images = evaluate_ocr_model(oc_model, val_dataloader, eval_steps = 10)
    oc_model.training_mode()
    log_oc_results(experiment = experiment, step = global_step, 
                   oc_logs = oc_logs, oc_imgs = oc_images, curves = {})
    target_oc_model = deepcopy(oc_model)
    for iteration in range(1, int(config.max_steps + 1) // config.sb3.n_steps):
        # Collect new rollout buffer
        oc_model.inference_mode(), agent.inference_mode()
        prev_global_step = global_step if global_step != 0 else -1
        for step in range(0, config.sb3.n_steps, config.num_envs):
            global_step += config.num_envs
            # TODO Check obs are in range [0; 1]
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
            next_value = agent.get_value(oc_model.get_slots(next_obs, training = False))
        rollout_buffer.finalize_tensors_calculate_and_store_GAE(last_done = next_done, last_value = next_value)

        # Train agent
        agent.training_mode()
        if config.sb3.train_feature_extractor:
            oc_model.training_mode()
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
            v_loss = ((newvalue - batch['return']) ** 2).mean()
            entropy_loss = -entropy.mean()
            alignment_loss = torch.Tensor([0]).to(device)
            if config.sb3.train_feature_extractor:
                with torch.no_grad():
                    target_slots = target_oc_model.get_slots(obs = start_obs, training = True)
                    gt_decoded = target_oc_model.decode_slots(obs = start_obs, slots = target_slots)
                slots = oc_model.get_slots(obs = start_obs, training = True)
                decoded = oc_model.decode_slots(obs = start_obs, slots = slots)
                alignment_loss = oc_model.get_oc_alignment_loss(gt_decoded = gt_decoded, decoded = decoded)
            loss = pg_loss + config.sb3.ent_coef * entropy_loss + config.sb3.vf_coef * v_loss + alignment_loss
            optimizer.optimizer_zero_grad()
            loss.backward()
            metrics.update(optimizer.optimizer_step('rl'))
            metrics.update({'ppo/value_loss': v_loss.item(), 'ppo/policy_gradient_loss': pg_loss.item(), 
                            'ppo/entropy_loss': entropy_loss.item(), 'ppo/alignment_loss': alignment_loss.item(),
                            'ppo/approx_kl': approx_kl.item(), 'ppo/old_approx_kl': old_approx_kl.item(), 
                            'ppo/clip_fraction': clipfracs})

        # Evaluation stage
        if (global_step // config.sb3.eval_freq - prev_global_step // config.sb3.eval_freq) != 0:
            oc_model.inference_mode(), agent.inference_mode()
            oc_logs, oc_images = evaluate_ocr_model(oc_model, val_dataloader, eval_steps = 10)
            path_to_video, mean_return = evaluate_agent(oc_model = oc_model, agent = agent, make_env_fns = eval_env_fns,
                                                        device = device, float_to_uint = float_to_uint)
            metrics.update({'eval/mean_return': mean_return})
            experiment.log_video(file = path_to_video, name = 'eval/video', step = global_step)
            log_oc_results(experiment = experiment, step = global_step, 
                           oc_logs = oc_logs, oc_imgs = oc_images, curves = ppg_curves)
            oc_model.training_mode(), agent.training_mode()
        
        # PPG stage (if needed)
        if config.sb3.train_feature_extractor and oc_model.requires_ppg() and (iteration % config.sb3.ppg_freq == 0):
            optimizer.optimizer_zero_grad()
            # TODO check that target models preserve their weights
            target_oc_model, target_agent = deepcopy(oc_model), deepcopy(agent)
            ppg_curves = {}

            for start_obs, future_obs in rollout_buffer.get_obs_generator(mode = 'ppg'):
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
            target_oc_model = deepcopy(oc_model)
            log_ppg_results(experiment, ppg_curves, step = global_step)

        y_true, y_pred = rollout_buffer.get_return_value()
        explained_variance = calculate_explained_variance(y_true, y_pred)
        rollout_buffer.reset_trajectories()
        metrics.update({'ppo/explained_variance': explained_variance, 
                        'ppo/steps_per_second': int(global_step / (time.time() - start_time))})
        experiment.log_metrics(metrics.convert_to_dict(), step = global_step)
        
if __name__ == "__main__":
    main()