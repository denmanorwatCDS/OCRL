import numpy as np

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