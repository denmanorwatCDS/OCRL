import torch
from torch import nn

class Alpha_MLP_Module(nn.Module):
    def __init__(self, ocr_rep_dim: int, num_slots: int, config: dict, num_stacked_obss: int=1) -> None:
        super(Alpha_MLP_Module, self).__init__()
        self.rep_dim = config.dims[-1]
        in_dim = ocr_rep_dim * num_stacked_obss
        
        # MLP
        net = []
        dims = config.dims.copy()
        dims[-1] = dims[-1] + 1
        for dim in dims:
            net.append(nn.Linear(in_dim, dim))
            net.append(nn.ReLU())
            in_dim = dim
        self._mlp = nn.Sequential(*net)

    def forward(self, state):
        preprocessed_slots_and_alphas = self._mlp(state)
        preproc_slots, preproc_alphas = torch.split(preprocessed_slots_and_alphas, [self.rep_dim, 1], dim = -1)
        preproc_alphas = torch.nn.functional.softmax(preproc_alphas, dim = 1)
        representation = torch.sum(preproc_slots * preproc_alphas, dim=1)
        return representation
