import numpy as np
import hydra
import omegaconf

from comet_ml import Experiment
from torch.utils.data import DataLoader
from pathlib import Path
from oc import ocrs
from oc.optimizer.optimizer import OCOptimizer
from data_utils.H5_dataset import H5Dataset

from utils.eval_tools import calculate_ari

def evaluate_model(model, val_dataloader):
    # OCR logging
    model.inference_mode()
    for j, batch in enumerate(val_dataloader):
        if j > 50:
            break
        mets = model.calculate_validation_data(batch['obss'].cuda())
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
                    attn_images['true_masks'].append(true_masks * 255)

        if 'masks' in batch.keys():
            orig_masks = batch['masks']
            fg_masks = (1 - orig_masks[:, -1])
            for name, mask in mets['masks']:
                mask, fg_mask = mask, mask * fg_masks
                if name not in ari_dict.keys():
                    ari_dict[name] = {'ari': [calculate_ari(orig_masks, mask)],
                                      'fg-ari': [calculate_ari(orig_masks, fg_mask, foreground = True)]}
                else:
                    ari_dict[name]['ari'].append(calculate_ari(orig_masks, mask))
                    ari_dict[name]['fg-ari'].append(calculate_ari(orig_masks, mask))

        for key in mets.keys():
            if key not in ['masks', 'masked_imgs', 'reconstructions']:
                if key not in precalc_data.keys():
                    precalc_data[key] = []
                precalc_data[key].append(mets[key])
                
    logs = {key: np.mean(val) for key, val in precalc_data.items()}
    for mask in ari_dict.keys():
        for metric in ari_dict[mask].keys():
            logs[f'{mask}: {metric}'] = np.mean(ari_dict[mask][metric])
    
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
    model.training_mode()
    return logs, imgs

@hydra.main(config_path="configs/", config_name="train_sb3")
def main(config):
    experiment = Experiment(
    api_key = config.comet.api_key,
    project_name = 'debug_oc_pretrain',
    workspace = config.comet.workspace
    )

    config.env['obs_size'] = config.ocr.obs_size
    # TODO train augmentations: shortest_size_resize (bicubic), hflip for train
    # TODO val augmentations: central_resize (bicubic for images, nearest neighbour for masks)

    if config.ocr['name'] == 'FT_DINOSAUR':
        import cv2
        from albumentations import ColorJitter, RandomResizedCrop, Rotate, HorizontalFlip, Compose
        random_flip = HorizontalFlip(p = 0.5)
        random_rotate = Rotate(limit = 15, border_mode = cv2.BORDER_REFLECT_101, p = 1.)
        random_transmute = RandomResizedCrop(height = config.env['obs_size'], width = config.env['obs_size'], 
                                             scale = (0.33, 1), ratio = (1/2, 2.), p = 1.)
        random_jitter = ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0., p = 1.)
        augmentations = [random_flip, random_rotate, random_transmute, random_jitter]
        augmenter = Compose(transforms = augmentations, p = 0.5)
        augment = lambda x: augmenter(image = x)['image']
    else:
        augment = lambda x: x

    min_val, max_val = config.ocr.image_limits[0], config.ocr.image_limits[1]
    uint_to_float = lambda x: (x.astype(np.float32) / 255 * (max_val - min_val) + min_val)
    
    # TODO add coco tests for FT-DINOSAUR
    
    """
    if '/coco_2017' in config.data_path:
        train_img_augs = Compose([RandomHorizontalFlip(p = 0.5), 
                                  ShortestSizeResize(size = config.env.obs_size, mode='bicubic'), 
                                  CenterCrop(size = config.env.obs_size),
                                  Renorm()])
        val_img_augs = Compose([ShortestSizeResize(size = config.env.obs_size, mode = 'bicubic'),
                                CenterCrop(size = config.env.obs_size),
                                Renorm()])
    
        val_mask_augs = Compose([ShortestSizeResize(size = config.env.obs_size, mode = 'nearest-exact'),
                                 CenterCrop(size = config.env.obs_size)])
        train_dataset = CoCoDataset(image_dir = config.data_path + '/train2017_img', mask_dir = None,
                                    img_transform = train_img_augs, mask_transform = None)
        val_dataset = CoCoDataset(image_dir = config.data_path + '/val2017_img', 
                                  mask_dir = config.data_path + '/val2017_instance',
                                  img_transform = val_img_augs, mask_transform = val_mask_augs)
        train_dataloader = DataLoader(train_dataset, batch_size = config.ocr.batch_size,
                                      shuffle = True, num_workers = 4, prefetch_factor = 1)
        val_dataloader = DataLoader(val_dataset, batch_size = config.ocr.batch_size,
                                    shuffle = False)
    else:
    """

    train_dataset = H5Dataset(datafile = config.data_path, uint_to_float = uint_to_float, augment = augment, is_train = True)
    val_dataset = H5Dataset(datafile = config.data_path, uint_to_float = uint_to_float, augment = None, is_train = False)
    train_dataloader = DataLoader(train_dataset, batch_size = config.ocr.batch_size,
                                  shuffle = True, num_workers = 4, prefetch_factor = 1)
    val_dataloader = DataLoader(val_dataset, batch_size = config.ocr.batch_size, shuffle = False)
    
    model = getattr(ocrs, config.ocr.name)(config.ocr, config.env)
    model = model.to('cuda')
    optimizer = OCOptimizer(omegaconf.OmegaConf.to_container(config.ocr.optimizer), oc_model = model, policy = None)
    i = 0
    model.training_mode()
    while i < config.max_steps:
        for batch in train_dataloader:
            batch = batch.to('cuda')
            optimizer.optimizer_zero_grad()
            loss, mets = model.get_loss(batch, do_dropout = False)
            loss.backward()
            mets.update(optimizer.optimizer_step('oc'))
            
            if i % 100 == 0:
                experiment.log_metrics(mets, step = i)

            if (i % 1_000 == 0):
                logs, imgs = evaluate_model(model = model, val_dataloader = val_dataloader)
                experiment.log_metrics({f'val/{key}': logs[key] for key in logs.keys()}, step = i)
                for key in imgs.keys():
                    experiment.log_image(image_data = imgs[key], name = f'val/{key}', 
                                         image_minmax = (0, 255), step = i)
            
            i += 1
            if i > config.max_steps:
                break
    path = Path.cwd()/'ft-dinosaur'
    model.save_oc_extractor(path)

if __name__ == "__main__":
    main()