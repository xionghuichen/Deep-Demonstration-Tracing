import math
import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    """Multiple Heads Attention
    """

    def __init__(
        self, query_dim, key_dim, value_dim, embed_dim, num_heads, dropout=0.0
    ):
        super(MultiheadAttention, self).__init__()
        self.mh_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
            kdim=key_dim,
            vdim=value_dim,
            dropout=dropout,
        )
        if query_dim != embed_dim:
            self.query_proj = nn.Linear(query_dim, embed_dim)
        else:
            self.query_proj = lambda x: x

    def forward(self, query, key, value):
        query = self.query_proj(query)
        context, attn_weights = self.mh_attn(query, key, value)
        return context, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len, 1)
        # check paper for equation
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2, )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x, cat=False):
        """
        x: [seq_len, batch_size, d_model]
        """
        if cat:
            x = torch.cat((x, self.pe[: x.size(0), :]), dim=-1)
        else:
            x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
