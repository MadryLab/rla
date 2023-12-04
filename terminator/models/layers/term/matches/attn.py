""" TERM Match Attention

This file includes modules which perform Attention to summarize the
information in TERM matches.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from terminator.models.layers.s2s_modules import (Normalize, PositionWiseFeedForward)

# pylint: disable=no-member


class TERMMatchAttention(nn.Module):
    """ TERM Match Attention

    A module with computes a node update using self-attention over
    all neighboring TERM residues and the edges connecting them.

    Attributes
    ----------
    W_Q : nn.Linear
        Projection matrix for querys
    W_K : nn.Linear
        Projection matrix for keys
    W_V : nn.Linear
        Projection matrix for values
    W_O : nn.Linear
        Output layer
    """
    def __init__(self, hparams):
        """
        Args
        ----
        hparams : dict
            Dictionary of model hparams (see :code:`~/scripts/models/train/default_hparams.json` for more info)
        """
        super().__init__()
        self.hparams = hparams
        hdim = hparams['term_hidden_dim']

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(hdim, hdim, bias=False)
        self.W_K = nn.Linear(hdim * 2, hdim, bias=False)
        self.W_V = nn.Linear(hdim * 2, hdim, bias=False)
        self.W_O = nn.Linear(hdim, hdim, bias=False)

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        mask_attn_dev = mask_attend.device
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).to(mask_attn_dev))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend.float() * attend
        return attend

    def forward(self, h_V, h_T, mask_attend=None):
        """ Self-attention update over residues in TERM matches

        Args
        ----
        h_V : torch.Tensor
            TERM match residues
            Shape: n_batch x sum_term_len x n_matches x n_hidden
        h_T : torch.Tensor
            Embedded structural features of target residue
            Shape: n_batch x sum_term_len x n_hidden
        mask_attend : torch.ByteTensor or None
            Mask for attention
            Shape: n_batch x sum_term_len # TODO: check shape

        Returns
        -------
        src_update : torch.Tensor
            TERM matches embedding update
            Shape: n_batch x sum_term_len x n_matches x n_hidden
        """
        n_batches, sum_term_len, n_matches = h_V.shape[:3]

        # append h_T onto h_V to form h_VT
        h_T_expand = h_T.unsqueeze(-2).expand(h_V.shape)
        h_VT = torch.cat([h_V, h_T_expand], dim=-1)
        query = h_V
        key = h_VT
        value = h_VT

        n_heads = self.hparams['matches_num_heads']
        num_hidden = self.hparams['term_hidden_dim']

        assert num_hidden % n_heads == 0

        d = num_hidden // n_heads
        Q = self.W_Q(query).view([n_batches, sum_term_len, n_matches, n_heads, d]).transpose(2, 3)
        K = self.W_K(key).view([n_batches, sum_term_len, n_matches, n_heads, d]).transpose(2, 3)
        V = self.W_V(value).view([n_batches, sum_term_len, n_matches, n_heads, d]).transpose(2, 3)

        attend_logits = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d)

        if mask_attend is not None:
            # we need to reshape the src key mask for residue-residue attention
            # expand to num_heads
            mask = mask_attend.unsqueeze(2).expand(-1, -1, n_heads, -1).unsqueeze(-1).float()
            mask_t = mask.transpose(-2, -1)
            # perform outer product
            mask = mask @ mask_t
            mask = mask.bool()
            # Masked softmax
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        src_update = torch.matmul(attend, V).transpose(2, 3).contiguous()
        src_update = src_update.view([n_batches, sum_term_len, n_matches, num_hidden])
        src_update = self.W_O(src_update)
        return src_update


class TERMMatchTransformerLayer(nn.Module):
    """ TERM Match Transformer Layer

    A TERM Match Transformer Layer that updates match embeddings via TERMMatchATtention

    Attributes
    ----------
    attention: TERMMatchAttention
        Transformer Attention mechanism
    dense: PositionWiseFeedForward
        Transformer position-wise FFN
    """
    def __init__(self, hparams):
        """
        Args
        ----
        hparams : dict
            Dictionary of model hparams (see :code:`~/scripts/models/train/default_hparams.json` for more info)
        """
        super().__init__()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['transformer_dropout'])
        hdim = hparams['term_hidden_dim']
        self.norm = nn.ModuleList([Normalize(hdim) for _ in range(2)])

        self.attention = TERMMatchAttention(hparams=self.hparams)
        self.dense = PositionWiseFeedForward(hdim, hdim * 4)

    def forward(self, src, target, src_mask=None, mask_attend=None, checkpoint=False):
        """ Apply one Transformer update to TERM matches

        Args
        ----
        src: torch.Tensor
            TERM Match features
            Shape: n_batch x sum_term_len x n_matches x n_hidden
        target: torch.Tensor
            Embedded structural features per TERM residue of target structure
            Shape: n_batch x sum_term_len x n_matches x n_hidden
        src_mask : torch.ByteTensor or None
            Mask for attention regarding TERM residues
            Shape : n_batch x sum_term_len
        mask_attend: torch.ByteTensor or None
            Mask for attention regarding matches
            Shape: n_batch x sum_term_len # TODO: check shape
        checkpoint : bool, default=False
            Whether to use gradient checkpointing to reduce memory usage

        Returns
        -------
        src: torch.Tensor
            Updated match embeddings
            Shape: n_batch x sum_term_len x n_matches x n_hidden
        """
        # Self-attention
        if checkpoint:
            dsrc = torch.utils.checkpoint.checkpoint(self.attention, src, target, mask_attend)
        else:
            dsrc = self.attention(src, target, mask_attend=mask_attend)
        src = self.norm[0](src + self.dropout(dsrc))

        # Position-wise feedforward
        if checkpoint:
            dsrc = torch.utils.checkpoint.checkpoint(self.dense, src)
        else:
            dsrc = self.dense(src)
        src = self.norm[1](src + self.dropout(dsrc))

        if src_mask is not None:
            src_mask = src_mask.unsqueeze(-1)
            src = src_mask * src
        return src


class TERMMatchTransformerEncoder(nn.Module):
    """ TERM Match Transformer Encoder

    A Transformer which uses a pool token to summarize the contents of TERM matches

    Attributes
    ----------
    W_v : nn.Linear
        Embedding layer for matches
    W_t : nn.Linear
        Embedding layer for target structure information
    W_pool: nn.Linear
        Embedding layer for pool token
    encoder_layers : nn.ModuleList of TERMMatchTransformerLayer
        Transformer layers for matches
    W_out : nn.Linear
        Output layer
    pool_token_init : nn.Parameter
        The embedding for the pool token used to gather information,
        reminiscent of [CLS] tokens in BERT
    """
    def __init__(self, hparams):
        """
        Args
        ----
        hparams : dict
            Dictionary of model hparams (see :code:`~/scripts/models/train/default_hparams.json` for more info)
        """
        super().__init__()
        self.hparams = hparams

        # Hyperparameters
        hidden_dim = hparams['term_hidden_dim']
        self.hidden_dim = hidden_dim
        num_encoder_layers = hparams['matches_layers']

        # Embedding layers
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_t = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_pool = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        layer = TERMMatchTransformerLayer

        # Encoder layers
        self.encoder_layers = nn.ModuleList([layer(hparams) for _ in range(num_encoder_layers)])

        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # lets try [CLS]-style pooling
        pool_token_init = torch.zeros(1, hidden_dim)
        torch.nn.init.xavier_uniform_(pool_token_init)
        self.pool_token = nn.Parameter(pool_token_init, requires_grad=True)

    def forward(self, V, target, mask):
        """ Summarize TERM matches

        Args
        ----
        V : torch.Tensor
            TERM Match embedding
            Shape: n_batches x sum_term_len x n_matches x n_hidden
        target : torch.Tensor
            Embedded structural information of target per TERM residue
            Shape: n_batches x sum_term_len x n_hidden
        mask : torch.ByteTensor
            Mask for TERM resides
            Shape: n_batches x sum_term_len

        Returns
        -------
        torch.Tensor
            Summarized TERM matches
            Shape: n_batches x sum_term_len x n_hidden
        """
        n_batches, sum_term_len = V.shape[:2]

        # embed each copy of the pool token with some information about the target ppoe
        pool = self.pool_token.view([1, 1, self.hidden_dim]).expand(n_batches, sum_term_len, -1)
        pool = torch.cat([pool, target], dim=-1)
        pool = self.W_pool(pool)
        pool = pool.unsqueeze(-2)

        V = torch.cat([pool, V], dim=-2)

        h_V = self.W_v(V)
        h_T = self.W_t(target)

        # Encoder is unmasked self-attention
        for _, layer in enumerate(self.encoder_layers):
            h_V = layer(h_V, h_T, mask.unsqueeze(-1).float(), checkpoint=self.hparams['gradient_checkpointing'])

        h_V = self.W_out(h_V)
        return h_V[:, :, 0, :]
