import numpy as np

def convert_single_channel_to_multi_channel(masks):
    concat_list = []
    for idx in range(4):
        concat_list.append(masks == idx)
    return np.expand_dims(np.stack(concat_list, axis = 0), axis = -1)