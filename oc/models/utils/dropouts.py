import torch
from torch import nn
import numpy as np

class PatchDropout(nn.Module):
    def __init__(self, apply_proba, min_patches_dropped, max_patches_dropped, patch_qty):
        super(PatchDropout, self).__init__()
        self.apply_proba = apply_proba
        self.min_patches_dropped, self.max_patches_dropped = min_patches_dropped, max_patches_dropped
        self.patch_qty = patch_qty
        self.do_dropout = True

    def forward(self, batch):
        apply_dropout = torch.rand(1)
        if self.do_dropout and self.max_patches_dropped > 1e-03 and apply_dropout < self.apply_proba:
            dropped_patches = torch.rand(1) * (self.max_patches_dropped - self.min_patches_dropped) + self.min_patches_dropped
            left_patches_qty = self.patch_qty - int(dropped_patches * self.patch_qty)
            batch_qty = batch.shape[0]
            batch_idxs = torch.arange(batch_qty).tolist()
            patch_idxs = [torch.multinomial(torch.ones(self.patch_qty), left_patches_qty, replacement = False).tolist() \
                          for i in range(len(batch_idxs))]
            new_batch = []
            for i in batch_idxs:
                new_batch.append(batch[i, patch_idxs[i]])
            new_batch = torch.stack(new_batch, axis = 0)
            return new_batch
        return batch

    def turn_off_dropout(self):
        self.do_dropout = False

    def turn_on_dropout(self):
        self.do_dropout = True


class CnnPatchDropout(nn.Module):
    def __init__(self, patch_dropout_proba, min_patches_dropped, max_patches_dropped, 
                       img_size, fmap_height, next_kernel_size,
                       device):
        super(CnnPatchDropout, self).__init__()
        self.patch_dropout_proba = patch_dropout_proba
        self.min_patches_dropped, self.max_patches_dropped = min_patches_dropped, max_patches_dropped

        img_height, img_width, self.kernel_height, self.kernel_width = img_size[0], img_size[1], next_kernel_size[0], next_kernel_size[1]
        self.height_patches_qty, self.width_patches_qty = (img_height // self.kernel_height), (img_width // self.kernel_width)
        leftout_height = img_size[0] - self.height_patches_qty * self.kernel_height
        leftout_width = img_size[1] - self.width_patches_qty * self.kernel_width
        self.low_height_pad, self.low_width_pad = leftout_height // 2, leftout_width // 2
        self.high_height_pad, self.high_width_pad = leftout_height - self.low_height_pad, leftout_width - self.low_width_pad

        self.height_patch_idx = np.arange(self.low_height_pad, img_height - self.high_height_pad, self.kernel_height)
        self.width_patch_idx = np.arange(self.low_width_pad, img_width - self.high_width_pad, self.kernel_width)
        self.patch_dropout_embed = nn.Parameter(data = 2 * (torch.rand((1, fmap_height, self.kernel_height, self.kernel_width))\
                                                            .expand(self.height_patches_qty * self.width_patches_qty, -1, -1, -1).to(device) - 0.5))
        
        self.fmap_height = fmap_height
        self.patch_qty = len(self.height_patch_idx) * len(self.width_patch_idx)
        self._latest_augmented_batch = None

    def forward(self, batch):
        if self.do_dropout:
            batch = batch.clone().detach()
            for img_idx in range(batch.shape[0]):
                apply_dropout = torch.rand(1)
                if self.max_patches_dropped > 1e-03 and apply_dropout < self.patch_dropout_proba:
                    dropped_patches_frac = torch.rand(1) * (self.max_patches_dropped - self.min_patches_dropped) + self.min_patches_dropped
                    dropped_patches = int(dropped_patches_frac * self.patch_qty)
                    to_swap = torch.from_numpy(np.random.choice([True if i < dropped_patches else False for i in range(self.patch_qty)], 
                                               size = self.patch_qty, replace=False))
                    to_swap = torch.Tensor(to_swap).to(batch.device)
                    
                    # Split image into rectangles of shape channels, patch_qty, patch_height, patch_width. Unfold, and slices, as well
                    # as simple indexing returns views, so this code modifies batch inplace
                    high_height_bound = -self.high_height_pad if self.high_height_pad != 0 else None
                    high_width_bound = -self.high_width_pad if self.high_width_pad != 0 else None
                    patched_img_from_batch = batch[img_idx, :, self.low_height_pad: high_height_bound, self.low_width_pad: high_width_bound].\
                        unfold(-1, self.kernel_width, self.kernel_width).unfold(-3, self.kernel_height, self.kernel_height)
                    patched_img_from_batch[:] = torch.where(to_swap.view(1, self.height_patches_qty, self.width_patches_qty, 1, 1), 
                                                 self.patch_dropout_embed.permute(1, 0, 2, 3).view(self.fmap_height, 
                                                                                                   self.height_patches_qty, self.width_patches_qty, 
                                                                                                   self.kernel_height, self.kernel_width), 
                                                 patched_img_from_batch)
            self._latest_augmented_batch = batch.clone().detach().cpu()
        return batch

    def turn_off_dropout(self):
        self.do_dropout = False

    def turn_on_dropout(self):
        self.do_dropout = True


class FeatureDropout(nn.Module):
    def __init__(self, feature_dropout_proba, max_features_dropped, min_features_dropped,
                       fmap_size, device):
        super(FeatureDropout, self).__init__()
        self.fmap_size = fmap_size
        self.feature_dropout_embed = nn.Parameter(data = 2 * (torch.rand(fmap_size).to(device) - 0.5))
        self.feature_dropout_proba = feature_dropout_proba
        self.max_features_dropped, self.min_features_dropped = max_features_dropped, min_features_dropped
        self.do_dropout = True

    def forward(self, features):
        if self.do_dropout:
            if self.max_features_dropped > 1e-03 and torch.rand(1) < self.feature_dropout_proba:
                dropped_features_frac = torch.rand(1) * (self.max_features_dropped - self.min_features_dropped) + self.min_features_dropped
                dropped_features = int(dropped_features_frac * self.fmap_size)
                to_swap = torch.from_numpy(np.random.choice([True if i < dropped_features else False for i in range(self.fmap_size)], 
                                                             size = self.fmap_size, replace = False))
                to_swap = torch.Tensor(to_swap)
                features[..., to_swap] = self.feature_dropout_embed[to_swap]
        return features
    
    def turn_off_dropout(self):
        self.do_dropout = True

    def turn_on_dropout(self):
        self.do_dropout = False

