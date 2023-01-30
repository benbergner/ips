import math

import torch
from torch import nn

def pos_enc_1d(D, len_seq):
    
    if D % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(D))
    pe = torch.zeros(len_seq, D)
    position = torch.arange(0, len_seq).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, D, 2, dtype=torch.float) *
                         -(math.log(10000.0) / D)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class ScaledDotProductAttention(nn.Module):
    ''' Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def compute_attn(self, q, k):
        
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(torch.softmax(attn, dim=-1))

        return attn

    def forward(self, q, k, v):
        
        attn = self.compute_attn(q, k)
        output = torch.matmul(attn, v)

        return output

class MultiHeadCrossAttention(nn.Module):
    ''' Multi-head cross-attention module '''

    def __init__(self, n_token, H, D, D_k, D_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        
        self.n_token = n_token
        self.H = H
        self.D_k = D_k
        self.D_v = D_v

        self.q = nn.Parameter(torch.empty((1, n_token, D)))
        q_init_val = math.sqrt(1 / D_k)
        nn.init.uniform_(self.q, a=-q_init_val, b=q_init_val)

        self.q_w = nn.Linear(D, H * D_k, bias=False)
        self.k_w = nn.Linear(D, H * D_k, bias=False)
        self.v_w = nn.Linear(D, H * D_v, bias=False)
        self.fc = nn.Linear(H * D_v, D, bias=False)

        self.attention = ScaledDotProductAttention(
            temperature=D_k ** 0.5,
            attn_dropout=attn_dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)

    def get_attn(self, x):
        
        D_k, H, n_token = self.D_k, self.H, self.n_token
        B, len_seq = x.shape[:2]

        q = self.q_w(self.q).view(1, n_token, H, D_k)
        k = self.k_w(x).view(B, len_seq, H, D_k)

        q, k = q.transpose(1, 2), k.transpose(1, 2)

        attn = self.attention.compute_attn(q, k)

        return attn

    def forward(self, x):
        
        D_k, D_v, H, n_token = self.D_k, self.D_v, self.H, self.n_token
        B, len_seq = x.shape[:2]

        # project and separate heads
        q = self.q_w(self.q).view(1, n_token, H, D_k)
        k = self.k_w(x).view(B, len_seq, H, D_k)
        v = self.v_w(x).view(B, len_seq, H, D_v)

        # transpose for attention dot product: B x H x len_seq x D_k or D_v
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # cross-attention
        x = self.attention(q, k, v)

        # transpose again: B x n_token x H x D_v
        # concat heads: B x n_token x (H * D_v)
        x = x.transpose(1, 2).contiguous().view(B, n_token, -1)
        # combine heads
        x = self.dropout(self.fc(x))
        # residual connection + layernorm
        x += self.q
        x = self.layer_norm(x)

        return x

class MLP(nn.Module):
    ''' MLP consisting of two feed-forward layers '''

    def __init__(self, D, D_inner, dropout=0.1):
        super().__init__()
        
        self.w_1 = nn.Linear(D, D_inner)
        self.w_2 = nn.Linear(D_inner, D)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(torch.relu(self.w_1(x)))
        x = self.dropout(x)
        
        x += residual
        x = self.layer_norm(x)

        return x

class Transformer(nn.Module):
    """ Cross-attention based transformer module """

    def __init__(self, n_token, H, D, D_k, D_v, D_inner, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        
        self.crs_attn = MultiHeadCrossAttention(n_token, H, D, D_k, D_v, attn_dropout=attn_dropout, dropout=dropout)
        self.mlp = MLP(D, D_inner, dropout=dropout)
    
    def get_scores(self, x):

        attn = self.crs_attn.get_attn(x)
        # Average scores over heads and tasks
        # Average over tasks is only required for multi-task learning (mnist).
        return attn.mean(dim=1).transpose(1, 2).mean(-1)

    def forward(self, x):

        return self.mlp(self.crs_attn(x))