import torch
from torch import nn
from oc.models.utils.networks import Conv2dBlock, conv2d
import torch.nn.functional as F
from oc.models.utils.networks import linear
import math

def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    eps = torch.finfo(logits.dtype).tiny
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau
    y_soft = F.softmax(gumbels, dim)
    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft
    
# Taken from https://github.com/singhgautam/slate/blob/master/train.py#L131-L146
def cosine_anneal(step, start_value, final_value, start_step, final_step):
    assert start_value >= final_value
    assert start_step <= final_step
    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b
    return value

class dVAE(nn.Module):
    def __init__(self, vocab_size, img_channels, img_size):
        super().__init__()

        kernel_size = 4
        self._encoder = nn.Sequential(
            Conv2dBlock(img_channels, 64, (kernel_size, kernel_size), kernel_size),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            conv2d(64, vocab_size, 1),
        )
        self._decoder = nn.Sequential(
            Conv2dBlock(vocab_size, 64, 1),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            conv2d(64, img_channels, 1),
        )

    def forward(self, obs, tau=1.0, hard=True):
        z_logits = F.log_softmax(self._encoder(obs), dim=1)
        z = gumbel_softmax(z_logits, tau, hard, dim=1)
        return z, z_logits

    def decode(self, z):
        return self._decoder(z)
    
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        T = input.shape[1]
        return self.dropout(input + self.pe[:, :T])
    
class TransformerDecoderBlock(nn.Module):
    def __init__(
        self, max_len, d_model, num_heads, dropout=0.0, gain=1.0, is_first=False
    ):
        super().__init__()

        self.is_first = is_first

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)

        mask = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1)
        self.self_attn_mask = nn.Parameter(mask, requires_grad=False)

        self.encoder_decoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_decoder_attn = MultiHeadAttention(
            d_model, num_heads, dropout, gain
        )

        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init="kaiming"),
            nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout),
        )

    def forward(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        T = input.shape[1]

        if self.is_first:
            input = self.self_attn_layer_norm(input)
            x = self.self_attn(input, input, input, self.self_attn_mask[:T, :T])
            input = input + x
        else:
            x = self.self_attn_layer_norm(input)
            x = self.self_attn(x, x, x, self.self_attn_mask[:T, :T])
            input = input + x

        x = self.encoder_decoder_attn_layer_norm(input)
        x = self.encoder_decoder_attn(x, encoder_output, encoder_output)
        input = input + x

        x = self.ffn_layer_norm(input)
        x = self.ffn(x)
        return input + x
        
class TransformerDecoder(nn.Module):
    def __init__(self, num_blocks, max_len, d_model, num_heads, dropout=0.0):
        super().__init__()

        if num_blocks > 0:
            gain = (3 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [
                    TransformerDecoderBlock(
                        max_len, d_model, num_heads, dropout, gain, is_first=True
                    )
                ]
                + [
                    TransformerDecoderBlock(
                        max_len, d_model, num_heads, dropout, gain, is_first=False
                    )
                    for _ in range(num_blocks - 1)
                ]
            )
        else:
            self.blocks = nn.ModuleList()

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        for block in self.blocks:
            input = block(input, encoder_output)

        return self.layer_norm(input)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, gain=1.0):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)

    def forward(self, q, k, v, attn_mask=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape

        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)

        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)
        return output