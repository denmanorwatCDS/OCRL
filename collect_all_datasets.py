import os
import h5py
import numpy as np
import hydra
from omegaconf import OmegaConf

num_tr = 1_000_000
num_val = 10_000
data_path = '/media/denis/data'

num_proc = 10
chunks = 50

def collect_data(config, data_path, img_path, obs_size, reset_after_each_step, env_type):
    print('Collecting data...')
    from envs.collect_data.collect_data_from_env import parallel_get_data
    config.obs_size = obs_size
    if env_type == 'Shapes':
        from envs import synthetic_envs as envs
        env = [getattr(envs, config.env)(config, seed=i) for i in range(num_proc)]

    elif env_type == 'Causal':
        from envs import cw_envs as envs
        env = [getattr(envs, config.env)(config, seed=i) for i in range(num_proc)]

    elif env_type == 'Pong':
        from envs import pong_envs as envs
        env = [getattr(envs, config.env)(config, seed=i) for i in range(num_proc)]
    
    file_name = f"{data_path}/data.hdf5"
    os.makedirs(data_path, exist_ok = True), os.makedirs(img_path, exist_ok = True)
    f = h5py.File(file_name, "w")
    tr_group = f.create_group("TrainingSet")
    tr_group.create_dataset(name = 'obss', shape = (num_tr, config.obs_size, config.obs_size, 3), 
                            chunks = (num_tr // chunks, config.obs_size, config.obs_size, 3), dtype = np.uint8)
    tr_group.create_dataset(name = 'dones', shape = (num_tr, ),
                            chunks = (num_tr // chunks), dtype = np.bool_)
    for i in range(chunks):
        obss, _, dones = parallel_get_data(env, env_type, num_proc, num_tr // chunks, 
                                                      reset_after_each_step = reset_after_each_step, render_masks = False,
                                                      img_path = img_path if i == 0 else None)
        tr_group["obss"][i * (num_tr // chunks): (i + 1) * (num_tr // chunks)] = obss
        tr_group["dones"][i * (num_tr // chunks): (i + 1) * (num_tr // chunks)] = dones
    val_group = f.create_group("ValidationSet")
    obss, masks, dones = parallel_get_data(env, env_type, num_proc, num_val, 
                                                      reset_after_each_step = reset_after_each_step, render_masks = True,
                                                      img_path = img_path)
    assert len(obss) == num_val
    permutation = np.random.permutation(len(obss))
    val_group["obss"], val_group["masks"] = np.stack(obss, axis = 0)[permutation], np.stack(masks, axis = 0)[permutation]
    print("done", os.getcwd(), file_name)
    f.close()

@hydra.main(config_path="configs/", config_name="collect_all")
def main(config):
    for obs_size in config.obs_sizes:
        for dict_key in config.envs.keys():
            name, env_config_yaml = config.envs[dict_key]['name'], config.envs[dict_key]['env_yaml']
            env_type, reset_after_each_step = config.envs[dict_key]['type'], config.envs[dict_key]['reset_after_each_step']
            dataset_folder  = f'{data_path}/{obs_size}x{obs_size}/{name}/dataset'
            examples_folder = f'{data_path}/{obs_size}x{obs_size}/{name}/examples'
            env_config = OmegaConf.load(f'../../../configs/env/{env_config_yaml}')
            default_confs = env_config.pop('defaults', None)
            if default_confs is not None:
                for i in range(len(default_confs)):
                    if default_confs[i] == '_self_':
                        del default_confs[i]
                assert len(default_confs) == 1, 'Expected that there is only one default config'
                agg_conf = default_confs[0]
                agg_conf = OmegaConf.load(f'../../../configs/env/{agg_conf}.yaml')
                agg_conf.merge_with(env_config)
            else:
                agg_conf = env_config
            collect_data(agg_conf, data_path = dataset_folder, img_path = examples_folder, 
                         obs_size = obs_size, reset_after_each_step = reset_after_each_step, env_type = env_type)

main()