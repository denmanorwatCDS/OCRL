import tqdm
import pybullet
import numpy as np
import multiprocessing
from envs.collect_data import collect_cw_utils, collect_sh_utils, collect_pong_utils

from PIL import Image

# Don't bother understand this code. It is just a mess of hacks to unify dataset collection
# from different environments each of which has its own unique interface and notion of objects 
# and etc ...

def prepare_env(env, env_type):
    if env_type == 'Causal':
        undercover_env = env.env.env.env
        if undercover_env._pybullet_client_w_o_goal_id is not None:
            client = undercover_env._pybullet_client_w_o_goal_id
        else:
            client = undercover_env._pybullet_client_full_id
        render_kwargs = {'width': env.observation_space.shape[0], 'height': env.observation_space.shape[1],
                         'viewMatrix': undercover_env.view_matrix, 'projectionMatrix': undercover_env.proj_matrix,
                         'renderer': pybullet.ER_BULLET_HARDWARE_OPENGL, 'physicsClientId': client}
    elif env_type == 'Pong':
        render_kwargs = None
    elif env_type == 'Shapes':
        render_kwargs = {'mode': 'mask'}
    return render_kwargs

def render_mask(env, env_type, render_kwargs):
    if env_type == 'Causal':
        _, _, image, _, mask = pybullet.getCameraImage(**render_kwargs)
        mask = collect_cw_utils.expand_background_class(mask)
        mask = collect_cw_utils.convert_single_channel_to_multi_channel(mask)
    elif env_type == 'Pong':
        mask = env.render_mask()
        mask = collect_pong_utils.convert_single_channel_to_multi_channel(mask)
    elif env_type == 'Shapes':
        mask = env.render(**render_kwargs)
    return mask

def fetch_number_of_masks_in_env(env, env_type):
    if env_type == 'Causal':
        return env.num_objects + 2
    
    elif env_type == 'Pong':
        return env._num_objects + 1
    
    elif env_type == 'Shapes':
        return env._num_objects + int(not env._wo_agent)

def get_data(seed, procidx, env, env_type, num_data, return_dict, render_masks, reset_after_each_step):
    np.random.seed(seed + procidx)
    obss, masks, dones = [], [], []
    env.reset()
    bar = tqdm.tqdm(total = num_data, smoothing = 0)
    render_kwargs = prepare_env(env, env_type)
    num_masks = fetch_number_of_masks_in_env(env = env, env_type = env_type)
    while len(obss) < num_data:
        image, _, done, info = env.step(env.action_space.sample())
        mask = None
        if render_masks:
            mask = render_mask(env = env, env_type = env_type, render_kwargs = render_kwargs)
        assert image.shape[-1] == 3, 'Assert that there are only three channels'

        obss.append(image)
        if mask is not None:
            masks.append(mask[0: num_masks, ...])
        dones.append(done)

        if reset_after_each_step:
            env.reset()
            dones[-1] = True
        elif done:
            env.reset()
        bar.update(1)
    dones[-1] = True
    return_dict[procidx] = {
        "obss": obss[:num_data],
        "masks": masks[:num_data],
        "dones": dones[:num_data]}
    
def parallel_get_data(env, env_type, num_proc, num_data, reset_after_each_step, img_path, render_masks = False):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    for i in range(num_proc):
        seed = np.random.randint(low=0, high=1e9)
        p = multiprocessing.Process(
            target=get_data, args=(seed, i, env[i], env_type, num_data // num_proc, return_dict, render_masks, reset_after_each_step)
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    obss, masks, dones = [], [], []
    for key, value in return_dict.items():
        obss.extend(value["obss"])
        dones.extend(value['dones'])
        if render_masks:
            masks.extend(value["masks"])
        if img_path is not None and not render_masks:
            for i in range(1):
                flag = value['dones'][i]
                Image.fromarray(value["obss"][i]).save(f'{img_path}/Example_{key}_{i}_done={flag}.png')
    return obss, masks, dones