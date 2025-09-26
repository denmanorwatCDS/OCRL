import torch
import torch.nn as nn
import torch.nn.functional as F

from oc.models.utils.networks import linear, gru_cell

class Queue():
    def __init__(self, size = 100):
        self.size = size
        self.features = []

    def add_batch(self, batch):
        random_idx = torch.multinomial(torch.ones(batch.shape[1]), num_samples = 25)
        random_batch = torch.multinomial(torch.ones(batch.shape[0]), num_samples = 10)
        features = batch[random_batch][:, random_idx].reshape(-1, batch.shape[-1])
        self.features.append(features)
        if len(self.features) > self.size:
            self.features.pop(0)
    
    def calculate_mean_std(self):
        if self.features:
            features = torch.cat(self.features, dim = 0)
            return torch.mean(features, dim = 0), torch.std(features, dim = 0)
        else:
            return torch.tensor(torch.nan), torch.tensor(torch.nan)


class SlotAttention(nn.Module):
    def __init__(
        self,
        num_iterations,
        num_slots,
        input_size,
        slot_size,
        mlp_hidden_size,
        heads,
        preinit_type,
        normalize_keys,
        epsilon=1e-8,
    ):
        super().__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.num_heads = heads
        self.preinit_type = preinit_type
        assert heads == 1, 'Only one head is supported now'

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        # Linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.key_normalizer = lambda x: x
        if normalize_keys:
            def key_normalizer(x):
                clipping_needed = torch.linalg.vector_norm(x, ord = 2, dim = -1, keepdim = True) > 1.
                return torch.where(clipping_needed, x / torch.linalg.vector_norm(x, ord = 2, dim = -1, keepdim = True),
                                   x)
            self.key_normalizer = key_normalizer
        self.project_v = linear(input_size, slot_size, bias=False)

        # Slot update functions.
        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init="kaiming"),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size),
        )
        self.feature_dist = Queue()

        # Parameters for Gaussian init (shared by all slots).
        if preinit_type == 'classic':
            self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
            nn.init.xavier_uniform_(self.slot_log_sigma)

        elif preinit_type == 'trainable':
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
            nn.init.xavier_uniform_(self.slot_log_sigma)
            self.initial_slots = nn.Parameter(torch.zeros(1, 1, slot_size))
            nn.init.xavier_uniform_(self.initial_slots)


    def forward(self, inputs, slots, save_all_slots = False):
        # `inputs` has shape [batch_size, num_inputs, input_size].
        # `slots` has shape [batch_size, num_slots, slot_size].
        all_slots = [slots.detach().cpu()]

        B, N_kv, D_inp = inputs.size()
        B, N_q, D_slot = slots.size()

        inputs = self.norm_inputs(inputs)
        k = (
            self.key_normalizer(self.project_k(inputs)).view(B, N_kv, self.num_heads, -1).transpose(1, 2)
        )  # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        v = (
            self.project_v(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)
        )  # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        k = ((self.slot_size // self.num_heads) ** (-0.5)) * k

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = (
                self.project_q(slots).view(B, N_q, self.num_heads, -1).transpose(1, 2)
            )  # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            attn_logits = torch.matmul(
                k, q.transpose(-1, -2)
            )  # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn = (
                F.softmax(
                    attn_logits.transpose(1, 2).reshape(B, N_kv, self.num_heads * N_q),
                    dim=-1,
                )
                .view(B, N_kv, self.num_heads, N_q)
                .transpose(1, 2)
            )  # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn_vis = attn.sum(1)  # Shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.matmul(
                attn.transpose(-1, -2), v
            )  # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            updates = updates.transpose(1, 2).reshape(
                B, N_q, -1
            )  # Shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(
                updates.view(-1, self.slot_size), slots_prev.view(-1, self.slot_size)
            )
            slots = slots.view(-1, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))
            all_slots.append(slots.detach().cpu())
        
        if not save_all_slots:
            del all_slots
            return slots, attn_vis
        
        return slots, attn_vis, all_slots
    
    def prepare_slots(self, x):
        B, *_ = x.size()
        
        if self.preinit_type == 'classic':
            init_slots = x.new_empty(B, self.num_slots, self.slot_size).normal_()
            init_slots = self.slot_mu + torch.exp(self.slot_log_sigma) * init_slots
        elif self.preinit_type == 'trainable':
            init_slots = self.initial_slots.repeat((B, 1, 1))
            init_slots = init_slots + torch.normal(mean = 0, std = 1, size = init_slots.shape).cuda()*\
                torch.exp(self.slot_log_sigma)
        elif self.preinit_type == 'statistics':
            init_slots = x.new_empty(B, self.num_slots, self.slot_size).normal_()
            mean, std = self.feature_dist.calculate_mean_std()
            init_slots = mean + init_slots * std
        return init_slots
    
    def update_statistics(self, batch):
        with torch.no_grad():
            batch_keys = self.key_normalizer(self.project_k(batch)).detach().clone()
            self.feature_dist.add_batch(batch_keys)

    def log_slot_mean_std(self):
        if self.preinit_type == 'classic':
            return self.slot_mu.detach(), torch.exp(self.slot_log_sigma).detach()
        elif self.preinit_type == 'trainable':
            return self.initial_slots.detach(), torch.exp(self.slot_log_sigma).detach()
        elif self.preinit_type == 'statistics':
            return self.feature_dist.calculate_mean_std()
        
    def log_feat_mean_std(self):
        return self.feature_dist.calculate_mean_std()
    
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if self.preinit_type == 'statistics':
            state['statistics'] = self.feature_dist
        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        if self.preinit_type == 'statistics':
            self.feature_dist = state_dict.pop('statistics')
        return super().load_state_dict(state_dict, *args, **kwargs)

class SlotAttentionModule(nn.Module):
    def __init__(
        self,
        num_iterations,
        num_slots,
        input_channels,
        slot_size,
        mlp_hidden_size,
        num_heads,
        preinit_type,
        normalize_keys,
        squash_features
    ):
        super().__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_channels = input_channels
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.squash_features = squash_features

        self.layer_norm = nn.LayerNorm(input_channels)
        self.mlp = nn.Sequential(
            linear(input_channels, input_channels, weight_init="kaiming"),
            nn.ReLU(),
            linear(input_channels, input_channels),
        )

        self.slot_attention = SlotAttention(
            num_iterations,
            num_slots,
            input_channels,
            slot_size,
            mlp_hidden_size,
            num_heads,
            preinit_type,
            normalize_keys = normalize_keys
        )

    def forward(self, input):
        # `image` has shape: [batch_size, img_channels, img_height, img_width].
        # `encoder_grid` has shape: [batch_size, pos_channels, enc_height, enc_width].
        x = self.mlp(self.layer_norm(input))
        if self.squash_features:
            x = torch.tanh(x)
        # `x` has shape: [batch_size, enc_height * enc_width, cnn_hidden_size].
        # Slot Attention module.
        slots = self.slot_attention.prepare_slots(input)
        slots, attn = self.slot_attention(x, slots)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # `attn` has shape: [batch_size, enc_height * enc_width, num_slots].

        return slots, attn
    
    def update_statistics(self, batch):
        self.slot_attention.update_statistics(batch)

    def log_slot_mean_std(self):
        return self.slot_attention.log_slot_mean_std()
        
    def log_feat_mean_std(self):
        return self.slot_attention.log_feat_mean_std()