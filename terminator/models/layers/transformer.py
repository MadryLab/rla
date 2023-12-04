import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .utils import gather_nodes
import math

class PoolAttentionLayer(nn.Module):
    def __init__(self, num_in, embed_dim):
        super().__init__()
        self.W_Q = nn.Linear(num_in, embed_dim, bias=True)
        self.W_O =nn.Linear(embed_dim, num_in, bias=True)
        self.W_KV = nn.Linear(num_in, embed_dim*2)
        self.embed_dim = embed_dim

    def forward(self, resid, x, x_mask):
        q = self.W_Q(resid) # N, H
        x = torch.cat([resid.unsqueeze(1), x], dim=1)
        dummy_mask = torch.ones(x_mask.shape[0]).to(x_mask.device).unsqueeze(1)
        x_mask = torch.cat([dummy_mask, x_mask], dim=1)
        kv = self.W_KV(x)
        k, v = kv.chunk(2, dim=-1) # N, T, H
        logits =  (k @ q.unsqueeze(-1)).squeeze(-1) # N, T
        logits[x_mask == 0] = -np.inf
        logits = logits/math.sqrt(self.embed_dim)
        attn =  F.softmax(logits, dim=-1)
        vals = (attn.unsqueeze(1) @ v).squeeze(1)
        out = self.W_O(vals)
        return out


class ConcatAttentionLayer(nn.Module):

    def __init__(self, num_in, num_e_in, embed_dim, out_dim, num_heads, attn_dropout, resid_dropout):
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.head_dim = embed_dim//num_heads
        
        self.W_Q = nn.Linear(num_in, embed_dim, bias=True)
        self.W_O = nn.Linear(embed_dim, out_dim, bias=True)

        self.pool_vec_proj = nn.Linear(num_in, num_e_in+num_in, bias=True)
        self.E_KV = nn.Linear(num_e_in+num_in, embed_dim*2, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)


    def _scaled_dot_product(self, q, k, v, e_mask):
        d_k = q.size()[-1]
        q_ = q.unsqueeze(-2)
        k_ = k.transpose(-2, -1)
        attn_logits = (q_ @ k_)
        if e_mask is not None:
            e_mask = e_mask.unsqueeze(2).unsqueeze(1)
            e_mask = e_mask.expand(attn_logits.shape)
            attn_logits[e_mask==0] = -np.inf
            # attn_logits = attn_logits.squeeze(-2) # N, C, T, K
            # attn_logits = attn_logits.permute(0, 2, 3, 1) # N, T, K, C
            # attn_logits[e_mask==0] = -np.inf
            # attn_logits = attn_logits.permute(0, 3, 1, 2) # N, C, T, K
            # attn_logits = attn_logits.unsqueeze(-2)

        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        attention = self.attn_drop(attention)
        values = attention @ v
        values = values.squeeze(-2)
        return values

    def forward(self, x, E_idx, E_features, e_mask, pool_vec=None):
        # x: N, T, C
        # E_idx: N, T, K
        # E_features: N, T, K, C_E
        # x_mask: N, T
        # e_mask: N, T, K
        batch_size, seq_length, _ = x.size()
        _, _, num_edges, _ = E_features.size()

        q = self.W_Q(x)
        x_gather = torch.cat([gather_nodes(x, E_idx), E_features], -1)
        if pool_vec is not None:
            pool_vec = self.pool_vec_proj(pool_vec).unsqueeze(1).unsqueeze(1)
            pool_vec = pool_vec.expand(-1, x_gather.shape[1], -1, -1)
            x_gather = torch.cat([pool_vec, x_gather], dim=2)
            e_mask_dummy = torch.ones(e_mask.shape[:2]).unsqueeze(-1).to(e_mask.device)
            e_mask = torch.cat([e_mask_dummy, e_mask], dim=2)
            num_edges = num_edges+1

        kv = self.E_KV(x_gather)
        k, v = kv.chunk(2, dim=-1) # N, T, K, *C
        # split out heads
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim) # N, T, H, C
        q = q.permute(0, 2, 1, 3) # [N, H, T, C]

        k = k.reshape(batch_size, seq_length, num_edges, self.num_heads, self.head_dim)
        k = k.permute(0, 3, 1, 2, 4) # N, H, T, K, C
        v = v.reshape(batch_size, seq_length, num_edges, self.num_heads, self.head_dim)
        v = v.permute(0, 3, 1, 2, 4) # N, H, T, K, C

        out = self._scaled_dot_product(q, k, v, e_mask) # N, H, T, C
        out = out.permute(0, 2, 1, 3) # N, T, H, C
        out = out.flatten(2, 3)
        out = self.W_O(out)
        out = self.resid_drop(out)
        return out 

class GraphEncoderLayer(nn.Module):
    def __init__(self, num_in, num_e_in, num_heads, 
                 embed_per_head, attn_dropout, 
                 dropout, mlp_multiplier):
        super().__init__()
        self.attn = ConcatAttentionLayer(
            num_in=num_in,
            num_e_in=num_e_in,
            embed_dim=embed_per_head*num_heads,
            out_dim=num_in,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=dropout,
        ) # already ends with a linear layer

        self.pool_attn = PoolAttentionLayer(num_in=num_in, embed_dim=embed_per_head)
        self.mlp = nn.Sequential(
            nn.Linear(num_in, mlp_multiplier*num_in),
            nn.GELU(), # gelu?
            nn.Linear(mlp_multiplier*num_in, num_in),
            nn.Dropout(dropout),

        )
        self.ln_1 = nn.LayerNorm(num_in)
        self.ln_2 = nn.LayerNorm(num_in)

    def run_mlp(self, x):
        x = self.ln_1(x)
        x = x + self.mlp(x)
        x = self.ln_2(x)
        return x

    def forward(self, x, E_idx, E_features, e_mask, x_mask, pool_vec=None):
        if pool_vec is not None:
            pool_vec = pool_vec + self.pool_attn(pool_vec, x, x_mask=x_mask) # update pool_vec
        x = x + self.attn(x=x, E_idx=E_idx, E_features=E_features, e_mask=e_mask, pool_vec=pool_vec)
        x = self.run_mlp(x)
        if pool_vec is not None:
            pool_vec = self.run_mlp(pool_vec)
        return x, pool_vec

class GraphTransformer(nn.Module):
    def __init__(
        self, num_in, num_e_in, num_heads, num_layers, embed_per_head,
        dropout, mlp_multiplier, num_out
    ):
        super().__init__()
        enc_layers = []
        for i in range(num_layers):
            enc_layer = GraphEncoderLayer(
                num_in=num_in,
                num_e_in=num_e_in,
                num_heads=num_heads,
                embed_per_head=embed_per_head,
                attn_dropout=dropout,
                dropout=dropout,
                mlp_multiplier=mlp_multiplier)
            enc_layers.append(enc_layer)
        self.enc_layers = nn.ModuleList(enc_layers)
        self.dropout = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(num_in)
        self.proj = nn.Linear(num_in, num_out)
        self.init_vec = nn.Parameter(torch.randn(num_in), requires_grad=True)

    def forward(self, x, E_idx, E_features, e_mask, x_mask):
        # Attention part
        pool_vec = self.init_vec.expand(x.shape[0], -1)
        x = self.dropout(x)
        for block in self.enc_layers:
            x, pool_vec = block(
                x=x, E_idx=E_idx, E_features=E_features, e_mask=e_mask, x_mask=x_mask,
                pool_vec=pool_vec,
            )
        x = self.ln_f(x)
        x = self.proj(x)
        return x