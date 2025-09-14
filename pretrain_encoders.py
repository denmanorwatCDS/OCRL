import numpy as np
import hydra
import omegaconf

from comet_ml import Experiment
from torch.utils.data import DataLoader
from pathlib import Path
from oc import ocrs
from oc.optimizer.optimizer import OCOptimizer
from data_utils.H5_dataset import H5Dataset

from utils.eval_tools import evaluate_ocr_model

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

    train_dataset = H5Dataset(datafile = config.data_path, uint_to_float = uint_to_float, augment = augment, is_train = True)
    val_dataset = H5Dataset(datafile = config.data_path, uint_to_float = uint_to_float, augment = None, is_train = False)
    train_dataloader = DataLoader(train_dataset, batch_size = config.ocr.batch_size,
                                  shuffle = True, num_workers = 4, prefetch_factor = 1, 
                                  drop_last = True)
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
            loss, mets = model.get_loss(batch, do_dropout = True)
            loss.backward()
            mets.update(optimizer.optimizer_step('oc'))
            
            if i % 100 == 0:
                experiment.log_metrics(mets, step = i)

            if (i % 1_000 == 0):
                logs, imgs = evaluate_ocr_model(model = model, val_dataloader = val_dataloader)
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