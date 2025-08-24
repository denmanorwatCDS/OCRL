import torch
import numpy as np
from sklearn.metrics import adjusted_rand_score

# reshape image for visualization
for_viz = lambda x: np.array(
    x.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255.0, dtype=np.uint8
)

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

    #save_attns_images(true_masks, 'True')
    #save_attns_images(pred_masks, 'Pred')
    #save_images(true_masks, pred_masks)
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