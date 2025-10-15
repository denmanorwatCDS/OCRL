import torch
from torch import nn
from copy import deepcopy

class OC_model(nn.Module):
    def __init__(self, ocr_config, obs_size, obs_channels):
        # Copy fields that needed in all models
        super(OC_model, self).__init__()
        self._obs_size = obs_size
        self._obs_channels = obs_channels
        self.num_slots = ocr_config.slotattr.num_slots
        self.rep_dim = ocr_config.slotattr.slot_size
        self.min_obs_bound, self.max_obs_bound = ocr_config.image_limits
    
    def _get_slots(self, obs, do_dropout, training):
        raise NotImplementedError

    def get_loss(self, training):
        raise NotImplementedError
    
    def get_grouped_parameters(self):
        raise NotImplementedError
    
    def get_paramwise_lr(self):
        paramwise_lr = {}
        for name, _ in self.get_grouped_parameters().items():
            paramwise_lr[name] = None
        return paramwise_lr
    
    def inference_mode(self):
        self.eval()
        for param in self.parameters():
            if param.dtype in (torch.float, torch.float16, torch.float32, torch.float64, 
                               torch.complex, torch.complex32, torch.complex64, torch.complex128):
                param.requires_grad = False

    def training_mode(self):
        self.train()
        for param in self.parameters():
            if param.dtype in (torch.float, torch.float16, torch.float32, torch.float64, 
                               torch.complex, torch.complex32, torch.complex64, torch.complex128):
                param.requires_grad = True

    def convert_attns_to_masks(self, obs, attns):
        attns = attns.reshape(
            obs.shape[0], self.num_slots, 1, self._obs_size, self._obs_size
        )
        obs = (deepcopy(obs).unsqueeze(1) - self.min_obs_bound) / (self.max_obs_bound - self.min_obs_bound)
        masked_imgs = obs * attns + (1.0 - attns)
        masked_imgs = masked_imgs * (self.max_obs_bound - self.min_obs_bound) + self.min_obs_bound
        return masked_imgs, attns
    
    def convert_tensor_to_img(self, obs):
        normalized_img = torch.clamp(((obs - self.min_obs_bound) / (self.max_obs_bound - self.min_obs_bound) * 255), 0, 255).to(torch.uint8)
        
        # Permute only last three axis
        permutation = [-len(normalized_img.shape) + i for i in range(len(normalized_img.shape))]
        permutation[-3:] = [-2, -1, -3]
        permutation = tuple(permutation)
        
        normalized_img = torch.permute(normalized_img, permutation)
        return normalized_img.cpu().numpy()
    
    def convert_attns_to_img(self, attns):
        normalized_img = torch.clamp(((attns - self.min_obs_bound) / (self.max_obs_bound - self.min_obs_bound) * 255), 0, 255).to(torch.uint8)

    def requires_ppg(self):
        return True

    def update_hidden_states(self):
        raise NotImplementedError