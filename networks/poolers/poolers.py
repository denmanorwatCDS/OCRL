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
    def __init__(self, obs_length, skill_length, nhead = 4, dim_feedforward = 64, num_layers = 2):
        super().__init__()
        self.projector = nn.Linear(obs_length, dim_feedforward)
        
        self.skill_token = nn.Parameter(data = torch.randn(size=[1, 1, dim_feedforward]) * 0.1)
        self.convert_to_embed = nn.Linear(obs_length, dim_feedforward, device='cuda:0')
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

        idx_tensor = torch.reshape(obj_idx, (batch_len, 1, 1))
        where_filter = torch.reshape(torch.arange(0, seq_len), shape=(1, seq_len, 1)).\
            expand(batch_len, -1, self.dim_feedforward).to('cuda')
        marked_seq = torch.where(where_filter == idx_tensor, processed_seq + self.skill_token, processed_seq)
        
        # Add mark (skill token) so pooler knows to which object skill is applied
        # Returns processed output token
        """
        seq = torch.cat((self.readout_token.expand(batch_len, -1, -1), seq), dim = 1)
        """
        transformed_seq = self.transformer_pooler(marked_seq)
        policy_data = transformed_seq[torch.arange(batch_len), obj_idx, :]
        return torch.cat([policy_data, skill], axis = -1)

class IdentityPooler(nn.Module):
    def __init__(self, obs_length, skill_length):
        super().__init__()
        self.outp_dim = obs_length + skill_length

    def forward(self, seq, skill, obj_idx = None):
        # Expecting seq to be of shape [Batch, Seq_len, obs_dim]
        return torch.cat((seq[:, 0], skill), dim=-1)
    
def get_pooler_network(name, obs_length, skill_length, pooler_config):
    if name == 'Transformer':
        return TransformerPooler(obs_length = obs_length, skill_length=skill_length, **pooler_config), True
    elif name == 'Fetcher':
        return FetcherPooler(obs_length = obs_length), False
    elif name == 'Identity':
        return IdentityPooler(obs_length = obs_length), False