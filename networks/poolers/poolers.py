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
        self.projector = nn.Linear(obs_length, dim_feedforward)
        self.convert_to_embed = nn.Linear(obs_length, dim_feedforward, device='cuda:0')
        _transformer_pooler = nn.TransformerEncoderLayer(dim_feedforward, nhead = nhead, dim_feedforward = dim_feedforward, 
                                                         batch_first = True)
        self.transformer_pooler = nn.TransformerEncoder(_transformer_pooler, num_layers = num_layers)
        self.outp_dim = dim_feedforward
        
    def forward(self, seq, obj_idx):
        # Expecting seq to be of shape [Batch, Seq_len, obs_dim]
        batch_len, seq_len, obs_dim = seq.shape
        # Add readout token
        target_objs = seq[torch.arange(0, seq.shape[0], 1), obj_idx]
        seq = self.projector(seq)
        # Add mark (skill token) so pooler knows to which object skill is applied
        # Returns processed output token
        readout_token = self.convert_to_embed(target_objs)
        seq = torch.cat((readout_token.unsqueeze(1), seq), dim = 1)
        policy_data = self.transformer_pooler(seq)[:, 0, :]
        return policy_data
    
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