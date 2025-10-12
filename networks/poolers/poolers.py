import torch
from torch import nn

class FetcherPooler(nn.Module):
    def __init__(self, obs_length):
        super().__init__()
        self.outp_dim = obs_length

    def forward(self, seq, obj_idx):
        # Expecting seq to be of shape [Batch, Seq_len, obs_dim]
        outp = seq[torch.arange(0, seq.shape[0], 1), obj_idx]
        return outp
    
class TransformerPooler(nn.Module):
    def __init__(self, obs_length, nhead = 4, dim_feedforward = 64, num_layers = 2):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(obs_length, dim_feedforward),
                                       nn.ReLU())
        _transformer_pooler = nn.TransformerEncoderLayer(dim_feedforward, nhead = nhead, dim_feedforward = dim_feedforward, 
                                                         batch_first = True)
        self.transformer_pooler = nn.TransformerEncoder(_transformer_pooler, num_layers = num_layers)
        self.outp_token = nn.Parameter(torch.randn((1, dim_feedforward)), requires_grad = True)
        self.skill_token = nn.Parameter(torch.randn((1, dim_feedforward)), requires_grad = True)
        self.outp_dim = dim_feedforward
        
    def forward(self, seq, obj_idx):
        # Expecting seq to be of shape [Batch, Seq_len, obs_dim]

        # Add readout token
        seq = self.projector(seq)
        seq = torch.cat((self.outp_token.expand((seq.shape[0], -1, -1)), seq), dim = 1)
        # Add mark (skill token) so pooler knows to which object skill is applied
        seq[torch.arange(0, seq.shape[0], 1), obj_idx] = seq[torch.arange(0, seq.shape[0], 1), obj_idx] +\
            self.skill_token
        # Returns processed output token
        pooled_data = self.transformer_pooler(seq)[:, 0, :]
        return pooled_data
    
class IdentityPooler(nn.Module):
    def __init__(self, obs_length):
        super().__init__()
        self.outp_dim = obs_length

    def forward(self, seq, obj_idx = None):
        # Expecting seq to be of shape [Batch, Seq_len, obs_dim]
        return seq[:, 0]
    
def get_pooler_network(name, obs_length, pooler_config):
    if name == 'Transformer':
        return TransformerPooler(obs_length = obs_length, **pooler_config), True
    elif name == 'Fetcher':
        return FetcherPooler(obs_length = obs_length), False
    elif name == 'Identity':
        return IdentityPooler(obs_length = obs_length), False