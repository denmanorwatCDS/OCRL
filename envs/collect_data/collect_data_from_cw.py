import multiprocessing
import pybullet
import tqdm
import numpy as np
from PIL import Image

def expand_background_class(masks):
    # Add to background class floor, bucket and table masks
    background_masks = [0, 2, 3]
    old_to_new_masks = {4: 2, 5: 3, 6: 4, 7: 5}
    for idx in background_masks:
        masks = np.where(masks == idx, np.zeros(masks.shape, dtype = masks.dtype), masks)
    for old_idx, new_idx in old_to_new_masks.items():
        masks = np.where(masks == old_idx, np.zeros(masks.shape, dtype = masks.dtype) + new_idx, masks)
    # Change masks so background mask is last
    masks = np.max(masks) - masks
    return masks

def convert_single_channel_to_multi_channel(masks):
    concat_list = []
    for idx in range(6):
        concat_list.append(masks == idx)
    return np.expand_dims(np.stack(concat_list, axis = 0), axis = -1)

def get_data(seed, procidx, env, num_data, return_dict, render_masks, reset_after_each_step):
    np.random.seed(seed + procidx)
    obss, masks, num_objs, labels = [], [], [], []
    
    env.reset()
    undercover_env = env.env.env.env
    if undercover_env._pybullet_client_w_o_goal_id is not None:
        client = undercover_env._pybullet_client_w_o_goal_id
    else:
        client = undercover_env._pybullet_client_full_id
    camera_params = {'width': env.observation_space.shape[0], 'height': env.observation_space.shape[1],
                     'viewMatrix': undercover_env.view_matrix, 'projectionMatrix': undercover_env.proj_matrix,
                     'renderer': pybullet.ER_BULLET_HARDWARE_OPENGL, 'physicsClientId': client}
    bar = tqdm.tqdm(total = num_data, smoothing = 0)
    while len(obss) < num_data:
        image, _, done, info = env.step(env.action_space.sample())
        num_ch, mask = image.shape[-1], None
        if render_masks:
            _, _, image, _, mask = pybullet.getCameraImage(**camera_params)
            mask = expand_background_class(mask)
            mask = convert_single_channel_to_multi_channel(mask)
            mask_ch = mask.shape[-1]
        
        image = image[..., :3]
        # every 3 channels is an image
        for i in range(num_ch // 3):
            obss.append(image[..., (i * 3): (i * 3) + 3])
            num_objs.append(env.num_objects)
            labels.append(env.target_obj_idx)
            if mask is not None:
                masks.append(mask[(i * (env.num_objects + 2)): (i * (env.num_objects + 2)) + (env.num_objects + 2), ...])
        if reset_after_each_step:
            env.reset()
        elif done:
            env.reset()
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