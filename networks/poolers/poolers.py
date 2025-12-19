import torch
from torch import nn
from math import sqrt

class FetcherPooler(nn.Module):
    def __init__(self, obs_length):
        super().__init__()
        self.outp_dim = obs_length

    def forward(self, seq, obj_idx):
        # Expecting seq to be of shape [Batch, Seq_len, obs_dim]
        outp = seq[torch.arange(0, seq.shape[0], 1), obj_idx]
        return outp
    
class TransformerPooler(nn.Module):
    def __init__(self, obs_length, skill_length, nhead = 4, dim_feedforward = 64, num_layers = 2):
        super().__init__()
        self.projector = nn.Linear(obs_length, dim_feedforward)
        self.q = nn.Sequential(nn.Linear(obs_length + skill_length, dim_feedforward)) 
        self.k, self.v = nn.Linear(dim_feedforward, dim_feedforward), nn.Linear(dim_feedforward, dim_feedforward)
        _transformer_pooler = nn.TransformerEncoderLayer(dim_feedforward, nhead = nhead, dim_feedforward = dim_feedforward, 
                                                         batch_first = True, norm_first = False)
        self.transformer_pooler = nn.TransformerEncoder(_transformer_pooler, num_layers = num_layers)
        """
        self.readout_token = nn.Parameter(torch.randn(size = (1, 1, dim_feedforward)) * 0.05)
        self.skill_token = nn.Parameter(torch.randn((1, 1, dim_feedforward)) * 1/3)
        """
        self.obs_dim = obs_length
        self.dim_feedforward = dim_feedforward
        self.outp_dim = dim_feedforward + skill_length
        
    def forward(self, seq, skill, obj_idx):
        # Expecting seq to be of shape [Batch, Seq_len, obs_dim]
        batch_len, seq_len, obs_dim = seq.shape
        # Add readout token
        processed_seq = self.projector(seq)
        
        # Add mark (skill token) so pooler knows to which object skill is applied
        # Returns processed output token
        """
        seq = torch.cat((self.readout_token.expand(batch_len, -1, -1), seq), dim = 1)
        """
        transformed_seq = self.transformer_pooler(processed_seq)
        query = torch.unsqueeze(self.q(torch.cat([seq[torch.arange(batch_len), obj_idx, :], skill], dim = -1)), dim = 1)
        keys, values = self.k(transformed_seq), self.v(transformed_seq)
        attention = torch.softmax(torch.sum(query * keys, axis=-1, keepdim=True)/sqrt(self.dim_feedforward), dim = -2)
        output = torch.sum(attention * values, axis=-2)
        return output

class IdentityPooler(nn.Module):
    def __init__(self, obs_length, skill_length):
        super().__init__()
        self.outp_dim = obs_length

    def forward(self, seq, skill, obj_idx = None):
        # Expecting seq to be of shape [Batch, Seq_len, obs_dim]
        return seq[:, 0]
    
class ConcatPooler(nn.Module):
    def __init__(self, obs_length, skill_length, obj_qty):
        super().__init__()
        self.outp_dim = obs_length * obj_qty + skill_length
        self.obj_embed = nn.Parameter(torch.randn(obs_length) * 0.02)

    def forward(self, seq, skill, obj_idx):
        batch_len, seq_len, obs_dim = seq.shape
        seq[torch.arange(0, batch_len), obj_idx] += self.obj_embed
        state_desc = torch.cat([seq[:, i] for i in range(seq_len)], axis=-1)
        embed = torch.cat([state_desc, skill], axis = -1)
        return embed
    
def get_pooler_network(name, obs_length, skill_length, pooler_config, obj_qty = None):
    if name == 'Transformer':
        return TransformerPooler(obs_length = obs_length, skill_length = skill_length, **pooler_config)
    elif name == 'Fetcher':
        return FetcherPooler(obs_length = obs_length)
    elif name == 'Identity':
        return IdentityPooler(obs_length = obs_length, skill_length = skill_length)
    elif name == 'Concat':
        return ConcatPooler(obs_length = obs_length, skill_length = skill_length, obj_qty = obj_qty)