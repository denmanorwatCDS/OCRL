import argparse, pathlib, omegaconf, comet_ml, functools, matplotlib, copy
from ReplayBuffers.path_replay_buffer import PathBuffer
from main_sac import fetch_config, make_env, prepare_batch
from RL.skill_model.metra_v2 import METRA
from eval_utils.eval_utils import StatisticsCalculator

def run():
    config = fetch_config()

    exp = comet_ml.start(project_name = 'metra')
    exp.log_parameters(config)

    make_seeded_env = functools.partial(make_env, env_name = config.env.name, 
                                        env_kwargs = config.env.env_kwargs,
                                        max_path_length = config.env.max_path_length,
                                        frame_stack = config.env.frame_stack, 
                                        normalizer_type = config.env.normalizer_type)
    env = make_seeded_env(seed = config.globals.seed, render_info = False)

    replay_buffer = PathBuffer(capacity_in_transitions = int(config.replay_buffer.common.max_transitions), 
                               batch_size = config.replay_buffer.common.batch_size, pixel_keys = {}, 
                               discount = config.replay_buffer.common.discount, 
                               gae_lambda = config.replay_buffer.discount, seed = config.globals.seed,
                               path_to_perfect_buffer = '/home/denis/Work/METRA_simplified/perfect_buffer.pickle')
    skill_model = METRA(obs_length = env.observation_space.shape[1], pooler_config = config.skill.slot_pooler, 
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
    skill_stats = StatisticsCalculator('skill')
    for i in range(10_000_000):
        batch = replay_buffer.sample_transitions()
        batch = prepare_batch(batch)
        logs, rewards = skill_model.train_components(observations = batch['observations'], 
                                                     next_observations = batch['next_observations'],
                                                     options = batch['options'],
                                                     obj_idxs = batch['obj_idxs'])
        skill_stats.save_iter(logs)
        if i % 10_000 == 0:
            logs = skill_stats.pop_statistics()
            exp.log_metrics(logs, step = i)

if __name__ == '__main__':
    matplotlib.use('Agg')
    run()