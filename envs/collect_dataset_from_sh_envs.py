import argparse
import multiprocessing
import os

import synthetic_envs
import gym
import h5py
import hydra
import numpy as np
import tqdm
from PIL import Image
import random

num_tr = 50_000
num_val = 1000
num_proc = 10

def get_data(procidx, env, num_data, return_dict):
    obss = []
    # objs = []
    num_objs = []
    labels = []
    # obs = env.reset()
    obs = env.reset()
    bar = tqdm.tqdm(total=num_data, smoothing=0)
    while len(obss) < num_data:
        obs, _, done, info = env.step(env.action_space.sample())
        num_ch = obs.shape[-1]
        # every 3 channels is an image
        for i in range(num_ch // 3):
            obss.append(obs[..., (i*3):(i*3)+3])
            Image.fromarray(obs[..., (i*3):(i*3)+3]).save("/home/denis/Work/OCRL_new/test{i}.jpg".format(i = random.randint(0, 100)), 
                                                          quality=100)
            num_objs.append(env._num_objects)
            labels.append(env._target_obj_idx)
        obs = env.reset()
        bar.update(1)
    return_dict[procidx] = {
        "obss": obss[:num_data],
        # "objs": objs[:num_data],
        "num_objs": num_objs[:num_data],
        "labels": labels[:num_data],
    }


def parallel_get_data(env, num_data):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    i = 0
    for i in range(num_proc):
        p = multiprocessing.Process(
            target=get_data, args=(i, env[i], num_data // num_proc, return_dict)
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    obss, objs, num_objs, labels = [], [], [], []
    for _, value in return_dict.items():
        obss.extend(value["obss"])
        # objs.extend(value["objs"])
        num_objs.extend(value["num_objs"])
        labels.extend(value["labels"])
    return obss, objs, num_objs, labels

# TODO fix precollection dataset. Now it samples from same environment, which is not good, at least on start
# "random-N5C4S4S2"
@hydra.main(config_path="../configs/env", config_name="odd-one-out-N4C2S2S1-oc")
def main(config):
    chunks, config.obs_size = 100, 128
    #config.wo_agent, config.agent_pos = True, None
    env = [getattr(synthetic_envs, config.env)(config, seed=i) for i in range(num_proc)]
    num_colors = len(config.COLORS)
    file_name = "/media/denis/data/OCRL_shapes/" + f"{config.env}-" f"N5C{num_colors}S1S1-Tr{num_tr}-Val{num_val}_{config.obs_size}.hdf5"
    f = h5py.File(file_name, "w")
    tr_group = f.create_group("TrainingSet")
    tr_group.create_dataset(name = 'obss', shape = (num_tr, config.obs_size, config.obs_size, 3), 
                            chunks = (num_tr // chunks, config.obs_size, config.obs_size, 3), dtype = np.uint8)
    tr_group.create_dataset(name = 'num_objs', shape = (num_tr, ),
                            chunks = (num_tr // chunks), dtype = np.uint8)
    tr_group.create_dataset(name = 'labels', shape = (num_tr, ),
                            chunks = (num_tr // chunks), dtype = np.uint8)
    for i in range(chunks):
        obss, objs, num_objs, labels = parallel_get_data(env, num_tr // chunks)
        # assert len(obss) == num_tr and len(labels) == num_tr and len(objs) == num_tr
        tr_group["obss"][i * (num_tr // chunks): (i + 1) * (num_tr // chunks)] = obss
        # tr_group["objs"] = objs
        tr_group["num_objs"][i * (num_tr // chunks): (i + 1) * (num_tr // chunks)] = num_objs
        tr_group["labels"][i * (num_tr // chunks): (i + 1) * (num_tr // chunks)] = labels
    val_group = f.create_group("ValidationSet")
    obss, objs, num_objs, labels = parallel_get_data(env, num_val)
    # assert len(obss) == num_val and len(labels) == num_val and len(objs) == num_val
    val_group["obss"] = obss
    # val_group["objs"] = objs
    val_group["num_objs"] = num_objs
    val_group["labels"] = labels
    print("done", os.getcwd(), file_name)
    f.close()


if __name__ == "__main__":
    main()
