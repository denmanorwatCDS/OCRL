import torch
from torch import nn

class Fetcher(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.outp_dim = obs_dim

    def forward(self, seq, obj_idx):
        return seq[torch.arange(0, seq.shape[0], 1), obj_idx]
    
class TransformerSlotAggregator(nn.Module):
    def __init__(self, obs_dim, nhead = 4, dim_feedforward = 1024, num_layers = 2):
        super().__init__()
        _transformer_pooler = nn.TransformerEncoderLayer(obs_dim, nhead = nhead, dim_feedforward = dim_feedforward, 
                                                         batch_first = True)
        self.transformer_pooler = nn.TransformerEncoder(_transformer_pooler, num_layers = num_layers)
        self.outp_token = nn.Parameter(torch.randn((1, obs_dim)), requires_grad = True)
        self.skill_token = nn.Parameter(torch.randn((1, obs_dim)), requires_grad = True)
        self.outp_dim = dim_feedforward
        
    def forward(self, seq, obj_idx):
        # Expecting [Batch, Seq_len, obs_dim]

        # Add readout token
        seq = torch.cat((self.outp_token.expand((seq.shape[0], -1, -1)), seq), dim = 1)
        # Add skill token
        seq[torch.arange(0, seq.shape[0], 1), obj_idx] = seq[torch.arange(0, seq.shape[0], 1), obj_idx] +\
            self.skill_token.expand((seq.shape[0], -1, -1))
        # Returns processed output token
        return self.transformer_pooler(seq)[:, 0, :]