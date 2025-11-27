import torch
import numpy as np
from sklearn.metrics import adjusted_rand_score
import cv2
import gym
import time
import os

# reshape image for visualization
for_viz = lambda x: np.array(
    x.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255.0, dtype=np.uint8
)

class Metrics():
    def __init__(self,):
        self.metric_array = {}
    
    def multiple_update(self, list_of_dicts):
        assert isinstance(list_of_dicts, list), 'Expected list of dicts'
        for dictionary in list_of_dicts:
            self.update(dictionary = dictionary)

    def update(self, dictionary):
        for key, value in dictionary.items():
            if key not in self.metric_array.keys():
                self.metric_array[key] = []
            logging_value = value
            if isinstance(logging_value, torch.Tensor):
                logging_value = logging_value.detach().cpu().numpy()
            self.metric_array[key].append(logging_value)
    
    def convert_to_dict(self):
        output_dict = {}
        for key in self.metric_array:
            group, name = key.split('/')
            output_dict['/'.join([group, 'mean_' + name])] = np.mean(self.metric_array[key])
            output_dict['/'.join([group, 'max_' + name])] = np.max(self.metric_array[key])
            output_dict['/'.join([group, 'last_' + name])] = self.metric_array[key][-1]
        self.metric_array = {}
        return output_dict
    
def add_prefix(dictionary, prefix):
    new_dictionary = {}
    for key in dictionary.keys():
        new_dictionary['/'.join([prefix, key])] = dictionary[key]
    del dictionary
    return new_dictionary

# Taken from https://github.com/singhgautam/slate/blob/master/slate.py
def visualize(images):
    _, H, W = images[0].shape  # first image is observation
    viz_imgs = []
    for _img in images:
        viz_imgs += list(torch.unbind(_img, dim=1))
    viz_imgs = torch.cat(viz_imgs, dim=-1)
    # return torch.cat(torch.unbind(viz_imgs,dim=0), dim=-2).unsqueeze(0)
    return viz_imgs

def get_item(x):
    if len(x.shape) == 0:
        return x.item()
    else:
        return x.detach().cpu().numpy()

def calculate_ari(true_masks, pred_masks, foreground=False):
    true_masks = true_masks.flatten(2)
    pred_masks = pred_masks.flatten(2)

    true_mask_ids = get_item(torch.argmax(true_masks, dim=1))
    pred_mask_ids = get_item(torch.argmax(pred_masks, dim=1))
    aris = []

    if foreground:
        for batch_idx in range(true_mask_ids.shape[0]):
            fg_true_mask_ids = true_mask_ids[batch_idx][true_mask_ids[batch_idx] != np.max(true_mask_ids)]
            fg_pred_mask_ids = pred_mask_ids[batch_idx][true_mask_ids[batch_idx] != np.max(true_mask_ids)]
            aris.append(adjusted_rand_score(fg_true_mask_ids, fg_pred_mask_ids))
    
    else:
        for batch_idx in range(true_mask_ids.shape[0]):
            aris.append(adjusted_rand_score(true_mask_ids[batch_idx], pred_mask_ids[batch_idx]))

    return aris

def evaluate_ocr_model(model, val_dataloader, full_eval = False):
    # OCR logging
    for j, batch in enumerate(val_dataloader):
        mets = model.calculate_validation_data(batch['obss'].cuda())
        if j > 250:
            break
        if j == 0:
            precalc_data, ari_dict, recon_images, attn_images = {}, {}, [], {}
            for key in mets['masks'].keys():
                attn_images[key] = []

        if j < 7:
            if 'reconstructions' in mets.keys():
                imgs = []
                for key, value in mets['reconstructions'].items():
                    imgs.append(value[0])
                imgs.append(model.convert_tensor_to_img(batch['obss'][0]))
                recon_images.append(np.concatenate(imgs, axis = 1))
            
            if 'masked_imgs' in mets.keys():
                for key, value in mets['masked_imgs'].items():
                    attn_images[key].append(value[0])

                if 'masks' in batch.keys():
                    true_masks = batch['masks'][0]
                    if 'true_masks' not in attn_images.keys():
                        attn_images['true_masks'] = []
                    attn_images['true_masks'].append(true_masks * 255)

        if ('masks' in batch.keys()) and full_eval:
            orig_masks = batch['masks'].to(torch.uint8)
            fg_masks = (1 - orig_masks[:, -2: -1])
            for name, mask in mets['masks'].items():
                mask = torch.permute(mask, (0, 1, 3, 4, 2))
                fg_mask = mask * fg_masks
                if name not in ari_dict.keys():
                    ari_dict[name] = {'ari': [calculate_ari(orig_masks, mask)],
                                      'fg-ari': [calculate_ari(orig_masks, fg_mask, foreground = True)]}
                else:
                    ari_dict[name]['ari'].append(calculate_ari(orig_masks, mask))
                    ari_dict[name]['fg-ari'].append(calculate_ari(orig_masks, mask, foreground=True))

        for key in mets.keys():
            if key not in ['masks', 'masked_imgs', 'reconstructions']:
                if key not in precalc_data.keys():
                    precalc_data[key] = []
                precalc_data[key].append(mets[key])
    
    logs = {key: np.mean(val) for key, val in precalc_data.items()}
    for mask in ari_dict.keys():
        for metric in ari_dict[mask].keys():
            logs[f'{mask}:{metric}'] = np.mean(ari_dict[mask][metric])
    
    if recon_images:
        imgs = {'observations': np.concatenate(recon_images, axis = 0)}

    max_width = -1
    for name, masks in attn_images.items():
        for elem in masks:
            max_width = max(max_width, elem.shape[0])
    
    for name, masks in attn_images.items():
        for i in range(len(masks)):
            attn_images[name][i] = np.pad(masks[i], ((0, max_width - masks[i].shape[0]), (0, 0), (0, 0), (0, 0)))
        attn_images[name] = np.concatenate(attn_images[name], axis = 1)
        attn_images[name] = np.transpose(attn_images[name], (1, 0, 2, 3))
        attn_images[name] = np.reshape(attn_images[name], (attn_images[name].shape[0], 
                                                           attn_images[name].shape[1] * attn_images[name].shape[2],
                                                           attn_images[name].shape[3]))
        imgs[name] = attn_images[name]
    return logs, imgs

def evaluate_agent(oc_model, agent, make_env_fns, device, float_to_uint, eval_episodes = 64):
    num_envs = len(make_env_fns)
    if num_envs == 1:
        eval_envs = gym.vector.SyncVectorEnv(make_env_fns)
    else:
        eval_envs = gym.vector.AsyncVectorEnv(make_env_fns, context = 'fork')
    per_episode_returns, total_dones = [], 0
    videos, current_returns = [[] for i in range(num_envs)], [0 for i in range(num_envs)]
    next_obs = torch.Tensor(eval_envs.reset()).to(device)
    black_screen = torch.permute(torch.zeros(next_obs[0].shape, dtype = torch.uint8), dims = (1, 2, 0)).to(device)
    with torch.no_grad():
        while total_dones < eval_episodes:
            slots = oc_model.get_slots(next_obs, training = False)
            action, *_ = agent.get_action_logprob_entropy(slots)
            next_obs, rewards, next_dones, _ = eval_envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(device)
            for i in range(num_envs):
                videos[i].append(torch.permute(float_to_uint(next_obs[i]), dims = (1, 2, 0)))
                current_returns[i] += rewards[i]
                if next_dones[i]:
                    total_dones += 1
                    per_episode_returns.append(current_returns[i])
                    current_returns[i] = 0
                    if total_dones >= eval_episodes:
                        for j in range(i + 1, num_envs):
                            videos[j].append(black_screen)
                        break
        video = []
        for i in range(num_envs):
            video.append(torch.stack(videos[i], dim = 0))
        video = torch.cat(video, dim=-2).cpu().numpy()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_filename = os.getcwd() + '/' + str(time.time()) + '.mp4'
        out = cv2.VideoWriter(video_filename, fourcc, 10, (video.shape[-2], video.shape[-3]))
        for frame in video:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        eval_envs.close()

        return video_filename, sum(per_episode_returns) / eval_episodes

def calculate_explained_variance(y_true, y_pred):
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

def log_ppg_results(experiment, step, 
                    logs_before_ppg, imgs_before_ppg, logs_after_ppg, imgs_after_ppg, curves):
    experiment.log_metrics({f'before_ppg/{key}': logs_before_ppg[key] for key in logs_before_ppg.keys()}, step = step)
    for key in imgs_before_ppg.keys():
        experiment.log_image(image_data = imgs_before_ppg[key], name = f'before_ppg/{key}', 
                             image_minmax = (0, 255), step = step)
        
    experiment.log_metrics({f'after_ppg/{key}': logs_after_ppg[key] for key in logs_after_ppg.keys()}, step = step)
    for key in imgs_after_ppg.keys():
        experiment.log_image(image_data = imgs_after_ppg[key], name = f'after_ppg/{key}', 
                             image_minmax = (0, 255), step = step)
        
    for key, value in curves.items():
        x = np.arange(len(curves[key]))
        experiment.log_curve(x = x, y = value, name = f'ppg/{key}', step = step)

def get_episodic_metrics(dones, infos):
    output = []
    if torch.any(dones):
        for i, info in enumerate(infos):
            if dones[i] and "episode" in info:
                output.append({'episode/return': info["episode"]["r"], 'episode/length': info["episode"]["l"],
                               'episode/mean_reward': info['episode']['r'] / info['episode']['l']})
    return output
