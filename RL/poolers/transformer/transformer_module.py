import math, torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, in_dim, d_model, nhead, num_layers, pos=None, norm_first=False):
        super(Transformer, self).__init__()
        self._cls_token = ClsToken(d_model)
        self._pos = pos
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead,
        )
        self._trans = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, state):
        # with cls token [batch_size, num_slots+1, d_model]
        state = torch.cat(
            [self._cls_token().repeat(state.shape[0], 1, 1), state], dim=1
        )
        # [batch_size, num_slots+1, d_model] -> [num_slots+1, batch_size, d_model]
        state = state.permute(1, 0, 2)
        state = state if self._pos is None else self._pos(state)
        return self._trans(state)[0]


class ClsToken(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self._cls_token = nn.Parameter(torch.normal(mean = torch.zeros(emb_size), 
                                                    std = torch.ones(emb_size)))

    def forward(self):
        return self._cls_token


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input):
        """
        input: seq_len x batch_size x d_model
        return: seq_len x batch_size x d_model
        """
        T = input.shape[0]
        return self.dropout(input + self.pe[:T])


# Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term) * 0.001
        pe[:, 0, 1::2] = torch.cos(position * div_term) * 0.001
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class StackedObsPositionalEncoding(nn.Module):
    def __init__(
        self,
        max_len,
        d_model,
        num_stacked_obss,
        include_initial_cls_token=False,
        dropout=0.0,
    ):
        super().__init__()

        if not include_initial_cls_token:
            assert max_len % num_stacked_obss == 0
            position = (
                torch.arange(max_len // num_stacked_obss)
                .repeat_interleave(num_stacked_obss)
                .unsqueeze(1)
            )
        else:
            assert (max_len - 1) % num_stacked_obss == 0
            position = torch.arange(
                (max_len - 1) // num_stacked_obss
            ).repeat_interleave(num_stacked_obss)
            position = torch.cat([torch.tensor([0]), position + 1], dim=0)
            position = position.unsqueeze(1)

        self.dropout = nn.Dropout(p=dropout)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term) * 0.001
        pe[:, 0, 1::2] = torch.cos(position * div_term) * 0.001
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

class SinusoidalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalEncoding, self).__init__()
        se = torch.zeros(max_len + 1, d_model)
        inp = torch.arange(0, max_len + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        se[:, 0::2] = torch.sin(inp * div_term)
        se[:, 1::2] = torch.cos(inp * div_term)

        se = se.unsqueeze(0).transpose(0, 1)
        self.register_buffer('se', se)

    def forward(self, x):
        return self.se[x, :]


class Transformer_Module(nn.Module):
    def __init__(self, ocr_rep_dim: int, ocr_num_slots: int, config: dict, num_stacked_obss: int=1) -> None:
        super(Transformer_Module, self).__init__()
        self.rep_dim = d_model = config.d_model
        self.config = config
        nhead = config.nhead
        num_layers = config.num_layers
        norm_first = config.norm_first

        # Positional encoding
        if num_stacked_obss > 1:
            pos = StackedObsPositionalEncoding(d_model=d_model, max_len=(ocr_num_slots*num_stacked_obss) + 1, num_stacked_obss=num_stacked_obss, include_initial_cls_token=True)
        elif config.pos_emb == "ape":  # Absolute Positional Embedding
            pos = PositionalEncoding(d_model=d_model, max_len=ocr_num_slots + 1)
        elif config.pos_emb == "lpe":  # Learnable Positional Embedding
            pos = PositionalEncoding(d_model=d_model, max_len=ocr_num_slots + 1)
        elif config.pos_emb == "None":
            pos = None

        if config.mlp_preprocessor:
            self.preprocessor = nn.Sequential(nn.Linear(ocr_rep_dim, 4 * ocr_rep_dim), nn.ReLU(), 
                                              nn.Linear(4 * ocr_rep_dim, ocr_rep_dim))
        else:
            self.preprocessor = nn.Identity()
        self._trans = Transformer(ocr_rep_dim, d_model, nhead, num_layers, pos, norm_first)

    def get_pos_emb(self, emb, x):
        x = (x + 1) / 2
        x = torch.clamp(x, 0.0, 1.0)
        x = (x // (1 / self.max_len)).long()
        x = emb(x)[..., -1, :]
        return x

    def forward(self, state):
        return self._trans(self.preprocessor(state))
