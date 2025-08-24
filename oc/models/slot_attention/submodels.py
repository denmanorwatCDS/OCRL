from torch import nn
from oc.models.utils.networks import Conv2dBlock, conv2d, PositionalEmbedding

class BroadCastDecoder(nn.Module):
    def __init__(self, obs_size, obs_channels, hidden_size, slot_size, initial_size):
        super().__init__()
        self._obs_size = obs_size
        self._initial_size = initial_size
        self._obs_channels = obs_channels
        self._decoder = nn.Sequential(
            Conv2dBlock(slot_size, hidden_size, kernel_size = 5, stride = 1, padding = 2),
            Conv2dBlock(hidden_size, hidden_size, kernel_size = 5, stride = 1, padding = 2),
            Conv2dBlock(hidden_size, hidden_size, kernel_size = 5, stride = 1, padding = 2),
            conv2d(hidden_size, obs_channels + 1, kernel_size = 3, stride = 1, padding = 1),
        )
        self._pos_emb = PositionalEmbedding(obs_size, slot_size)

    def _spatial_broadcast(self, slots):
        slots = slots.unsqueeze(-1).unsqueeze(-1)
        return slots.repeat(1, 1, self._obs_size, self._obs_size)

    def forward(self, slots):
        B, N, _ = slots.shape
        # [batch_size * num_slots, d_slots]
        slots = slots.flatten(0, 1)
        # [batch_size * num_slots, d_slots, obs_size, obs_size]
        slots = self._spatial_broadcast(slots)
        out = self._decoder(self._pos_emb(slots))
        img_slots, masks = out[:, :self._obs_channels], out[:, -1:]
        img_slots = img_slots.view(
            B, N, self._obs_channels, self._obs_size, self._obs_size
        )
        masks = masks.view(B, N, 1, self._obs_size, self._obs_size)
        masks = masks.softmax(dim=1)
        recon_slots_masked = img_slots * masks
        return recon_slots_masked.sum(dim=1), masks