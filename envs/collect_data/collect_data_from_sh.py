import multiprocessing
import tqdm
from PIL import Image
import numpy as np

def get_data(seed, procidx, env, num_data, return_dict, render_masks, reset_after_each_step):
    np.random.seed(seed + procidx)
    obss, masks, num_objs, labels = [], [], [], []
    obs = env.reset()
    bar = tqdm.tqdm(total = num_data, smoothing = 0)
    while len(obss) < num_data:
        obs, _, done, info = env.step(env.action_space.sample())
        if render_masks:
            mask = env.render(mode = 'mask')
        num_ch = obs.shape[-1]
        # every 3 channels is an image
        for i in range(num_ch // 3):
            obss.append(obs[..., (i * 3): (i * 3) + 3])
            if render_masks:
                shape_number = env._num_objects + int(not env._wo_agent)
                masks.append(mask[(i * (shape_number + 1)): (i * (shape_number + 1)) + (shape_number + 1), ...])
            num_objs.append(env._num_objects)
            labels.append(env._target_obj_idx)
        if reset_after_each_step:
            obs = env.reset()
        elif done:
            obs = env.reset()
        bar.update(1)
    return_dict[procidx] = {
        "obss": obss[:num_data],
        "masks": masks[:num_data],
        "num_objs": num_objs[:num_data],
        "labels": labels[:num_data],
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
        labels.extend(value["labels"])
        if render_masks:
            masks.extend(value["masks"])
        if img_path is not None:
            for i in range(10):
                Image.fromarray(value["obss"][i]).save(f'{img_path}/Example_{key}_{i}.png')
    return obss, masks, num_objs, labels