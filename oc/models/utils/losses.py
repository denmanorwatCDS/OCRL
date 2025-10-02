import torch
import numpy as np
from scipy import optimize

def similarity_func(slots1, slots2, tau):
    scalar_product = torch.sum(slots1 * slots2, axis=-1)
    norm_product = torch.linalg.norm(slots1, axis = -1, ord = 2) *\
        torch.linalg.norm(slots2, axis = -1, ord = 2)
    similarity = torch.exp((scalar_product / (norm_product + 1e-5)) / tau)
    return similarity

def frame_consistency_loss(matched_starting_slots, matched_future_slots, tau):
    batch_shape = matched_starting_slots.shape[0]
    matched_starting_slots = matched_starting_slots[torch.arange(batch_shape), torch.arange(batch_shape)]
    matched_future_slots = matched_future_slots[torch.arange(batch_shape), torch.arange(batch_shape)]
    matched_starting_slots = torch.unsqueeze(matched_starting_slots, dim = 1)
    matched_future_slots = torch.unsqueeze(matched_future_slots, dim = 2)
    similarity = similarity_func(matched_starting_slots, matched_future_slots, tau)
    positive_scores = torch.diagonal(similarity, dim1 = -2, dim2 = -1)
    negative_scores = torch.sum(similarity, dim = 1) - positive_scores
    frame_contrastive = -torch.log(torch.mean(positive_scores / (negative_scores + 1e-5)))
    return frame_contrastive

def time_loss(matched_starting_slots, matched_future_slots, tau):
    # It is of shape batch x batch
    similarity = similarity_func(matched_starting_slots, matched_future_slots, tau)
    similarity = torch.mean(similarity, axis=-1)
    positive_scores = torch.diagonal(similarity).clone()
    negative_scores = torch.sum(similarity, dim = 1) - positive_scores
    timestep_contrastive = -torch.log(torch.mean(positive_scores / (negative_scores + 1e-5)))
    return timestep_contrastive

def hungarian_loss(starting_slots, future_slots, tau = 0.1):
    batch_shape, slot_qty, slot_size = starting_slots.shape[0], starting_slots.shape[1], starting_slots.shape[2]
    # Initial shape: batch x slots x slot_size
    # Add second batch dimension
    # I.e. all tensors are now: batch x batch x slots x slot_size
    starting_slots = torch.unsqueeze(starting_slots, axis = 1).repeat(1, batch_shape, 1, 1)
    future_slots = torch.unsqueeze(future_slots, axis = 0).repeat(batch_shape, 1, 1, 1)
    # Add second slot dimension
    # I.e. all tensors are now: batch x batch x (slots x 1 | 1 x slots) x slot_size
    starting_slots_ = torch.unsqueeze(starting_slots, axis = 3).detach()
    future_slots_ = torch.unsqueeze(future_slots, axis = 2).detach()
    # Now, costs tensor is of shape: batch x batch x slots x slots
    costs = similarity_func(starting_slots_, future_slots_, tau).cpu().numpy()
    matched_idxs = np.full((batch_shape, batch_shape, 2, slot_qty, slot_size), np.nan)
    for image_1_idx in range(batch_shape):
        for image_2_idx in range(batch_shape):
            cost_matrix = costs[image_1_idx][image_2_idx]
            row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
            row_ind = np.repeat(np.expand_dims(row_ind, axis = 1), repeats = slot_size, axis = 1)
            col_ind = np.repeat(np.expand_dims(col_ind, axis = 1), repeats = slot_size, axis = 1)
            matched_idxs[image_1_idx][image_2_idx] = np.stack([row_ind, col_ind], axis = 0)
    matched_idxs = torch.from_numpy(matched_idxs).to(starting_slots.device).to(torch.long)
    permuted_starting_slots = torch.gather(starting_slots, dim = 2, index = matched_idxs[:, :, 0])
    permuted_future_slots = torch.gather(future_slots, dim = 2, index = matched_idxs[:, :, 1])

    return time_loss(permuted_starting_slots, permuted_future_slots, tau) +\
        frame_consistency_loss(permuted_starting_slots, permuted_future_slots, tau)