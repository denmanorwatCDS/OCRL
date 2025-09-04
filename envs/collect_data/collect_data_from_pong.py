import multiprocessing
import tqdm
import numpy as np
from PIL import Image

def convert_single_channel_to_multi_channel(masks):
    concat_list = []
    for idx in range(4):
        concat_list.append(masks == idx)
    return np.expand_dims(np.stack(concat_list, axis = 0), axis = -1)

def get_data(seed, procidx, env, num_data, return_dict, render_masks, reset_after_each_step = False):
    np.random.seed(seed + procidx)
    obss, masks, num_objs, labels = [], [], [], []
    obs = env.reset()
    bar = tqdm.tqdm(total = num_data, smoothing = 0)
    while len(obss) < num_data:
        obs, _, done, info = env.step(env.action_space.sample())
        mask = None
        if render_masks:
            mask = env.render_mask()
            mask = convert_single_channel_to_multi_channel(mask)
        num_ch = obs.shape[-1]
        # every 3 channels is an image
        for i in range(num_ch // 3):
            obss.append(obs[..., (i * 3): (i * 3) + 3])
            num_objs.append(env._num_objects)
            if mask is not None:
                masks.append(mask[(i * (env._num_objects + 1)): (i * (env._num_objects + 1)) + (env._num_objects + 1), ...])
        if reset_after_each_step:
            obs = env.reset()
        elif done:
            obs = env.reset()
        bar.update(1)
    return_dict[procidx] = {
        "obss": obss[:num_data],
        "masks": masks[:num_data],
        "num_objs": num_objs[:num_data],
        #"labels": labels[:num_data],
    }


def parallel_get_data(env, num_proc, num_data, reset_after_each_step, img_path, render_masks = False):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    i = 0
    for i in range(num_proc):
        seed = np.random.randint(low=0, high=1e9)
        p = multiprocessing.Process(
            target=get_data, args=(seed, i, env[i], num_data // num_proc, return_dict, render_masks, reset_after_each_step)
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    obss, masks, num_objs, labels = [], [], [], []
    for key, value in return_dict.items():
        obss.extend(value["obss"])
        num_objs.extend(value["num_objs"])
        #labels.extend(value["labels"])
        if render_masks:
            masks.extend(value["masks"])
        if img_path is not None:
            for i in range(10):
                Image.fromarray(value["obss"][i]).save(f'{img_path}/Example_{key}_{i}.png')
    return obss, masks, num_objs, labels