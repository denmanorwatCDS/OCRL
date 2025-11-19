import hydra
import torch
import omegaconf
import os, sys, random
import random

from comet_ml import Experiment
from torch.utils.data import DataLoader
from oc import ocrs
from oc.optimizer.optimizer import OCOptimizer
from data_utils.H5_dataset import H5Dataset

from utils.eval_tools import evaluate_ocr_model
from utils.train_tools import get_model_name, get_uint_to_float

import numpy as np

@hydra.main(config_path="configs/", config_name="train_ocr")
def main(config):
    PATH_TO_PRETRAIN_ENCODERS_DOT_PY = sys.path[0]
    path_to_target_dataset = '/'.join([config.dataset.root_path, 
                                       f"{config.dataset.obs_size}x{config.dataset.obs_size}", 
                                       config.dataset.name, "dataset", "data.hdf5"])
    path_to_dir, path_to_model = get_model_name(config.dataset.name, 
                                                f"{config.dataset.obs_size}x{config.dataset.obs_size}", 
                                                config.ocr.name, config.save_name)
    os.makedirs('/'.join([PATH_TO_PRETRAIN_ENCODERS_DOT_PY, path_to_dir]), exist_ok=True)
    model_save_path = '/'.join([PATH_TO_PRETRAIN_ENCODERS_DOT_PY, path_to_model])
    
    experiment = Experiment(
    api_key = config.comet.api_key,
    project_name = config.comet.project,
    workspace = config.comet.workspace
    )

    # TODO train augmentations: shortest_size_resize (bicubic), hflip for train
    # TODO val augmentations: central_resize (bicubic for images, nearest neighbour for masks)

    if config.ocr['name'] == 'FT_DINOSAUR':
        import cv2
        from albumentations import ColorJitter, RandomResizedCrop, Rotate, HorizontalFlip, Compose
        random_flip = HorizontalFlip(p = 0.5)
        random_rotate = Rotate(limit = 15, border_mode = cv2.BORDER_REFLECT_101, p = 1.)
        random_transmute = RandomResizedCrop(height = config.ocr.obs_size, width = config.ocr.obs_size, 
                                             scale = (0.33, 1), ratio = (1/2, 2.), p = 1.)
        random_jitter = ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0., p = 1.)
        augmentations = [random_flip, random_rotate, random_transmute, random_jitter]
        augmenter = Compose(transforms = augmentations, p = 0.5)
        augment = lambda x: augmenter(image = x)['image']
    else:
        augment = lambda x: x

    uint_to_float = get_uint_to_float(config.ocr.image_limits[0], config.ocr.image_limits[1])

    train_dataset = H5Dataset(datafile = path_to_target_dataset, uint_to_float = uint_to_float, 
                              use_future = config.ocr.slotattr.matching_loss.use, 
                              future_steps = config.ocr.slotattr.matching_loss.steps_into_future,
                              augment = augment, is_train = True, debug = config.debug)
    val_dataset = H5Dataset(datafile = path_to_target_dataset, uint_to_float = uint_to_float,
                            use_future = False, 
                            future_steps = config.ocr.slotattr.matching_loss.steps_into_future, 
                            augment = None, is_train = False)
    train_dataloader = DataLoader(train_dataset, batch_size = config.ocr.batch_size,
                                  shuffle = True, num_workers = config.num_workers, prefetch_factor = config.prefetch_factor, 
                                  drop_last = True)
    val_dataloader = DataLoader(val_dataset, batch_size = config.ocr.batch_size, shuffle = False)
    
    model = getattr(ocrs, config.ocr.name)(config.ocr, obs_size = config.dataset.obs_size, 
                                           obs_channels = config.dataset.obs_channels)
    model = model.to('cuda')
    optimizer = OCOptimizer(omegaconf.OmegaConf.to_container(config.ocr.optimizer), oc_model = model, policy = None)
    i = 0
    best_loss, best_model, best_idx = 1e9, None, None
    model.training_mode()
    while i < config.max_steps:
        for batch, future_batch in train_dataloader:
            batch, future_batch = batch.cuda(), future_batch.cuda()
            optimizer.optimizer_zero_grad()
            loss, mets = model.get_loss(batch, future_batch, do_dropout = True)
            loss.backward()
            mets.update(optimizer.optimizer_step('oc'))
            
            if i % 100 == 0:
                experiment.log_metrics(mets, step = i)
            
            if (i % 10_000 == 0):
                model.inference_mode()
                if (i % 30_000 == 0):
                    logs, imgs = evaluate_ocr_model(model = model, val_dataloader = val_dataloader, full_eval = True)
                else:
                    logs, imgs = evaluate_ocr_model(model = model, val_dataloader = val_dataloader)
                experiment.log_metrics({f'val/{key}': logs[key] for key in logs.keys()}, step = i)
                for key in imgs.keys():
                    experiment.log_image(image_data = imgs[key], name = f'val/{key}', 
                                         image_minmax = (0, 255), step = i)
                model.training_mode()
                if best_loss > logs['total_loss']:
                    best_loss, best_model, best_idx = logs['total_loss'], model.state_dict(), i
                    if i >= config.max_steps - 10_000:
                        torch.save(best_model, model_save_path + f';step:{best_idx}')
                
            i += 1
            if i > config.max_steps:
                break

if __name__ == "__main__":
    main()