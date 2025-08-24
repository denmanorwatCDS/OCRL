import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
    weight_init="xavier",
):
    m = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
    )
    if weight_init == "kaiming":
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.zeros_(m.bias)
    return m

def deconv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
    weight_init="xavier",
):
    m = nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
    )
    if weight_init == "kaiming":
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.zeros_(m.bias)
    return m

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.m = conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=True,
            weight_init="kaiming",
        )

    def forward(self, x):
        x = self.m(x)
        return F.relu(x)
    
class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride = 1, padding = 0, output_padding = 0):
        super().__init__()
        self.m = deconv2d(in_channels, out_channels, kernel_size, stride, padding, 
                          output_padding=output_padding, bias = True, weight_init = "kaiming")
        
    def forward(self, x):
        x = self.m(x)
        return F.relu(x)

def linear(in_features, out_features, bias=True, weight_init="xavier", gain=1.0):
    m = nn.Linear(in_features, out_features, bias)
    if weight_init == "kaiming":
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    if bias:
        nn.init.zeros_(m.bias)
    return m


def gru_cell(input_size, hidden_size, bias=True):
    m = nn.GRUCell(input_size, hidden_size, bias)
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    return m


class CNNEncoder(nn.Module):
    def __init__(self, obs_channels, hidden_size):
        super().__init__()
        self.kernel_size = (5, 5)
        self.hidden_size = hidden_size
        self._encoder = nn.Sequential(
            Conv2dBlock(obs_channels, self.hidden_size, self.kernel_size, 1, 2),
            Conv2dBlock(self.hidden_size, self.hidden_size, self.kernel_size, 1, 2),
            Conv2dBlock(self.hidden_size, self.hidden_size, self.kernel_size, 1, 2),
            conv2d(self.hidden_size, self.hidden_size, self.kernel_size, 1, 2)
        )

    def forward(self, obs):
        features = self._encoder(obs)
        return features
    

class PositionalEmbedding(nn.Module):
    def __init__(self, obs_size: int, obs_channels: int):
        super().__init__()
        width = height = obs_size
        east = torch.linspace(0, 1, width).repeat(height)
        west = torch.linspace(1, 0, width).repeat(height)
        south = torch.linspace(0, 1, height).repeat(width)
        north = torch.linspace(1, 0, height).repeat(width)
        east = east.reshape(height, width)
        west = west.reshape(height, width)
        south = south.reshape(width, height).T
        north = north.reshape(width, height).T
        # (4, h, w)
        linear_pos_embedding = torch.stack([north, south, west, east], dim=0)
        linear_pos_embedding.unsqueeze_(0)  # for batch size
        self.channels_map = nn.Conv2d(4, obs_channels, kernel_size=1)
        self.register_buffer("linear_position_embedding", linear_pos_embedding)

    def forward(self, x):
        bs_linear_position_embedding = self.linear_position_embedding.expand(
            x.size(0), 4, x.size(2), x.size(3)
        )
        x = x + self.channels_map(bs_linear_position_embedding)
        return x
    