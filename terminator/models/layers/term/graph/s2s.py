""" TERM MPNN modules

This file contains Attention and Message Passing implementations
of the TERM MPNN. """

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from terminator.models.layers.s2s_modules import (Normalize, PositionWiseFeedForward)
from terminator.models.layers.utils import (gather_term_nodes, cat_term_neighbors_nodes, cat_term_edge_endpoints,
                                            merge_duplicate_term_edges)

# pylint: disable=no-member


class TERMNeighborAttention(nn.Module):
    """ TERM Neighbor Attention

    A module which computes a node update using self-attention over
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
    def __init__(self, num_hidden, num_in, num_heads=4):
        """
        Args
        ----
        num_hidden : int
            Hidden dimension, and dimensionality of querys
        num_in : int
            Dimensionality of keys and values
        num_heads : int, default=4
            Number of heads to use in Attention
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax

        Args
        ----
        attend_logits : torch.Tensor
            Attention logits
        mask_attend: torch.ByteTensor
            Mask on Attention logits
        dim : int, default=-1
            Dimension to perform softmax along

        Returns
        -------
        attend : torch.Tensor
            Softmaxed :code:`attend_logits`
        """
        negative_inf = np.finfo(np.float32).min
        mask_attn_dev = mask_attend.device
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).to(mask_attn_dev))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend.float() * attend
        return attend

    def forward(self, h_V, h_EV, mask_attend=None):
        """ Self-attention update over nodes of a TERM graph

        Args
        ----
        h_V: torch.Tensor
            Central node features
            Shape: n_batch x n_terms x n_nodes x n_hidden
        h_EV: torch.Tensor
            Neighbor features, which includes the node vector concatenated onto
            the edge connecting the central node to the neighbor node
            Shape: n_batch x n_terms x n_nodes x n_neighbors x n_in
        mask_attend: torch.ByteTensor or None
            Mask for attention regarding neighbors
            Shape: n_batch x n_terms x n_nodes x k

        Returns
        -------
        h_V_update: torch.Tensor
            Node embedding update
            Shape: n_batch x n_terms x n_nodes x n_hidden
        """
        # Queries, Keys, Values
        n_batch, n_terms, n_nodes, n_neighbors = h_EV.shape[:4]
        n_heads = self.num_heads

        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_batch, n_terms, n_nodes, 1, n_heads, 1, d])
        K = self.W_K(h_EV).view([n_batch, n_terms, n_nodes, n_neighbors, n_heads, d, 1])
        V = self.W_V(h_EV).view([n_batch, n_terms, n_nodes, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K).view([n_batch, n_terms, n_nodes, n_neighbors, n_heads]).transpose(-2, -1)
        attend_logits = attend_logits / np.sqrt(d)

        if mask_attend is not None:
            mask = mask_attend.unsqueeze(3).expand(-1, -1, -1, n_heads, -1)
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        # Attentive reduction
        h_V_update = torch.matmul(attend.unsqueeze(-2), V.transpose(3, 4))
        h_V_update = h_V_update.view([n_batch, n_terms, n_nodes, self.num_hidden])
        h_V_update = self.W_O(h_V_update)
        return h_V_update


class TERMNodeTransformerLayer(nn.Module):
    """ TERM Node Transformer Layer

    A TERM Node Transformer Layer that updates nodes via TERMNeighborAttention

    Attributes
    ----------
    attention: TERMNeighborAttention
        Transformer Attention mechanism
    dense: PositionWiseFeedForward
        Transformer position-wise FFN
    """
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0.1):
        """
        Args
        ----
        num_hidden : int
            Hidden dimension, and dimensionality of querys in TERMNeighborAttention
        num_in : int
            Dimensionality of keys and values
        num_heads : int, default=4
            Number of heads to use in TERMNeighborAttention
        dropout : float, default=0.1
            Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.attention = TERMNeighborAttention(num_hidden, num_in, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_VE, mask_V=None, mask_attend=None):
        """ Apply one Transformer update on nodes in a TERM graph

        Args
        ----
        h_V: torch.Tensor
            Central node features
            Shape: n_batch x n_terms x n_nodes x n_hidden
        h_VE: torch.Tensor
            Neighbor features, which includes the node vector concatenated onto
            the edge connecting the central node to the neighbor node
            Shape: n_batch x n_terms x n_nodes x n_neighbors x n_in
        mask_V : torch.ByteTensor or None
            Mask for attention regarding TERM residues
            Shape : n_batch x n_terms x n_nodes
        mask_attend: torch.ByteTensor or None
            Mask for attention regarding neighbors
            Shape: n_batch x n_terms x n_nodes x k

        Returns
        -------
        h_V: torch.Tensor
            Updated node embeddings
            Shape: n_batch x n_terms x n_nodes x n_hidden
        """
        # Self-attention
        dh = self.attention(h_V, h_VE, mask_attend)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # Apply node mask
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class TERMEdgeEndpointAttention(nn.Module):
    """ TERM Edge Endpoint Attention

    A module which computes an edge update using self-attention over
    all edges that it share a 'home residue' with, as well as the nodes
    that form those edges.

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
    def __init__(self, num_hidden, num_in, num_heads=4):
        """
        Args
        ----
        num_hidden : int
            Hidden dimension, and dimensionality of querys
        num_in : int
            Dimensionality of keys and values
        num_heads : int, default=4
            Number of heads to use in Attention
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax

        Args
        ----
        attend_logits : torch.Tensor
            Attention logits
        mask_attend: torch.ByteTensor
            Mask on Attention logits
        dim : int, default=-1
            Dimension to perform softmax along

        Returns
        -------
        attend : torch.Tensor
            Softmaxed :code:`attend_logits`
        """
        negative_inf = np.finfo(np.float32).min
        mask_attn_dev = mask_attend.device
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).to(mask_attn_dev))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend.float() * attend
        return attend

    def forward(self, h_E, h_EV, E_idx, mask_attend=None):
        """ Self-attention update over edges in a TERM graph

        Args
        ----
        h_E: torch.Tensor
            Edge features in kNN dense form
            Shape: n_batch x n_terms x n_nodes x k x n_hidden
        h_EV: torch.Tensor
            'Neighbor' edge features, or all edges which share a 'central residue' with that edge,
            as well as the node features for both nodes that compose that edge.
            Shape: n_batch x n_terms x n_nodes x k x n_in
        mask_attend: torch.ByteTensor or None
            Mask for attention regarding neighbors
            Shape: n_batch x n_terms x n_nodes x k

        Returns
        -------
        h_E_update: torch.Tensor
            Update for edge embeddings
            Shape: n_batch x n_terms x n_nodes x k x n_hidden
        """
        # Queries, Keys, Values
        n_batch, n_terms, n_aa, n_neighbors = h_E.shape[:-1]
        n_heads = self.num_heads

        assert self.num_hidden % n_heads == 0

        d = self.num_hidden // n_heads
        Q = self.W_Q(h_E).view([n_batch, n_terms, n_aa, n_neighbors, n_heads, d]).transpose(3, 4)
        K = self.W_K(h_EV).view([n_batch, n_terms, n_aa, n_neighbors, n_heads, d]).transpose(3, 4)
        V = self.W_V(h_EV).view([n_batch, n_terms, n_aa, n_neighbors, n_heads, d]).transpose(3, 4)

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d)

        if mask_attend is not None:
            # we need to reshape the src key mask for edge-edge attention
            # expand to num_heads
            mask = mask_attend.unsqueeze(3).expand(-1, -1, -1, n_heads, -1).unsqueeze(-1).double()
            mask_t = mask.transpose(-2, -1)
            # perform outer product
            mask = mask @ mask_t
            mask = mask.bool()
            # Masked softmax
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        # Attentive reduction
        h_E_update = torch.matmul(attend, V).transpose(3, 4).contiguous()
        h_E_update = h_E_update.view([n_batch, n_terms, n_aa, n_neighbors, self.num_hidden])
        h_E_update = self.W_O(h_E_update)
        # nondirected edges are actually represented as two directed edges in opposite directions
        # to allow information flow, merge these duplicate edges
        h_E_update = merge_duplicate_term_edges(h_E_update, E_idx)
        return h_E_update


class TERMEdgeTransformerLayer(nn.Module):
    """ TERM Edge Transformer Layer

    A TERM Edge Transformer Layer that updates edges via TERMEdgeEndpointAttention

    Attributes
    ----------
    attention: TERMEdgeEndpointAttention
        Transformer Attention mechanism
    dense: PositionWiseFeedForward
        Transformer position-wise FFN
    """
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0.1):
        """
        Args
        ----
        num_hidden : int
            Hidden dimension, and dimensionality of querys in TERMNeighborAttention
        num_in : int
            Dimensionality of keys and values
        num_heads : int, default=4
            Number of heads to use in TERMNeighborAttention
        dropout : float, default=0.1
            Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.attention = TERMEdgeEndpointAttention(num_hidden, num_in, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_E, h_EV, E_idx, mask_E=None, mask_attend=None):
        """ Apply one Transformer update on edges in a TERM graph

        Args
        ----
        h_E: torch.Tensor
            Edge features in kNN dense form
            Shape: n_batch x n_terms x n_nodes x k x n_hidden
        h_EV: torch.Tensor
            'Neighbor' edge features, or all edges which share a 'central residue' with that edge,
            as well as the node features for both nodes that compose that edge.
            Shape: n_batch x n_terms x n_nodes x k x n_in
        mask_E : torch.ByteTensor or None
            Mask for attention regarding TERM edges
            Shape : n_batch x n_terms x n_nodes
        mask_attend: torch.ByteTensor or None
            Mask for attention regarding 'neighbor' edges
            Shape: n_batch x n_terms x n_nodes x k

        Returns
        -------
        h_E: torch.Tensor
            Updated edge embeddings
            Shape: n_batch x n_terms x n_nodes x k x n_hidden
        """
        # Self-attention
        dh = self.attention(h_E, h_EV, E_idx, mask_attend)
        h_E = self.norm[0](h_E + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_E)
        h_E = self.norm[1](h_E + self.dropout(dh))

        if mask_E is not None:
            mask_E = mask_E.unsqueeze(-1)
            h_E = mask_E * h_E
        return h_E


class TERMNodeMPNNLayer(nn.Module):
    """ TERM Node MPNN Layer

    A TERM Node MPNN Layer that updates nodes via generating messages and feeding the update
    through a feedforward network

    Attributes
    ----------
    W1, W2, W3: nn.Linear
        Layers for message computation
    dense: PositionWiseFeedForward
        Transformer position-wise FFN
    """

    # pylint: disable=unused-argument
    # num_heads is not used, but exists for compatibility with options for the Attention equivalent
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=None):
        """
        Args
        ----
        num_hidden : int
            Hidden dimension, and dimensionality of querys in TERMNeighborAttention
        num_in : int
            Dimensionality of keys and values
        num_heads : int, default=4
            Number of heads to use in TERMNeighborAttention
        dropout : float, default=0.1
            Dropout rate
        scale : int or None, default=None
            Scaling integer by which to divde the sum of computed messages.
            If None, the mean of the messages will be used instead.
        """
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Apply one MPNN update on nodes in a TERM graph

        Args
        ----
        h_V: torch.Tensor
            Central node features
            Shape: n_batch x n_terms x n_nodes x n_hidden
        h_VE: torch.Tensor
            Neighbor features, which includes the node vector concatenated onto
            the edge connecting the central node to the neighbor node
            Shape: n_batch x n_terms x n_nodes x n_neighbors x n_in
        mask_V : torch.ByteTensor or None
            Mask for message-passing regarding TERM residues
            Shape : n_batch x n_terms x n_nodes
        mask_attend: torch.ByteTensor or None
            Mask for message-passing regarding neighbors
            Shape: n_batch x n_terms x n_nodes x k

        Returns
        -------
        h_V: torch.Tensor
            Updated node embeddings
            Shape: n_batch x n_terms x n_nodes x n_hidden
        """
        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(F.relu(self.W2(F.relu(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        # note: this inherently decreases the magnitudes of messages for smaller TERMs
        # that wasn't intentional, but maybe that's a good thing
        if self.scale is None:
            dh = torch.mean(h_message, dim=-2)
        else:
            dh = torch.sum(h_message, dim=-2) / self.scale

        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class TERMEdgeMPNNLayer(nn.Module):
    """ TERM Edge MPNN Layer

    A TERM Edge MPNN Layer that updates edges via generating messages and feeding the update
    through a feedforward network

    Attributes
    ----------
    W1, W2, W3: nn.Linear
        Layers for message computation
    dense: PositionWiseFeedForward
        Transformer position-wise FFN
    """

    # pylint: disable=unused-argument
    # num_heads is not used, but exists for compatibility with options for the Attention equivalent
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None):
        """
        Args
        ----
        num_hidden : int
            Hidden dimension, and dimensionality of querys in TERMNeighborAttention
        num_in : int
            Dimensionality of keys and values
        num_heads : int, default=4
            Number of heads to use in TERMNeighborAttention
        dropout : float, default=0.1
            Dropout rate
        """
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.W1 = nn.Linear(num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_E, h_EV, E_idx, mask_E=None, mask_attend=None):
        """ Apply one MPNN update on edges in a TERM graph

        Args
        ----
        h_E: torch.Tensor
            Edge features in kNN dense form
            Shape: n_batch x n_terms x n_nodes x k x n_hidden
        h_EV: torch.Tensor
            'Neighbor' edge features, or all edges which share a 'central residue' with that edge,
            as well as the node features for both nodes that compose that edge.
            Shape: n_batch x n_terms x n_nodes x k x n_in
        mask_E : torch.ByteTensor or None
            Mask for message-passing regarding TERM edges
            Shape : n_batch x n_terms x n_nodes
        mask_attend: torch.ByteTensor or None
            Mask for message-passing regarding 'neighbor' edges
            Shape: n_batch x n_terms x n_nodes x k

        Returns
        -------
        h_E: torch.Tensor
            Updated edge embeddings
            Shape: n_batch x n_terms x n_nodes x k x n_hidden
        """

        dh = self.W3(F.relu(self.W2(F.relu(self.W1(h_EV)))))
        dh = merge_duplicate_term_edges(dh, E_idx)
        if mask_attend is not None:
            dh = mask_attend.unsqueeze(-1) * dh

        h_E = self.norm[0](h_E + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_E)
        h_E = self.norm[1](h_E + self.dropout(dh))

        if mask_E is not None:
            mask_E = mask_E.unsqueeze(-1)
            h_E = mask_E * h_E
        return h_E


class TERMGraphTransformerEncoder(nn.Module):
    """ TERM Graph Transformer Encoder

    Alternating node and edge update layers to update the represenation of TERM graphs

    Attributes
    ----------
    W_v : nn.Linear
        Embedding layer for nodes
    W_e : nn.Linear
        Embedding layer for edges
    node_encoder : nn.ModuleList of TERMNodeTransformerLayer or TERMNodeMPNNLayer
        Update layers for nodes
    edge_encoder : nn.ModuleList of TERMEdgeTransformerLayer or TERMEdgeMPNNLayer
        Update layers for edges
    W_out : nn.Linear
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
        node_features = hparams['term_hidden_dim']
        edge_features = hparams['term_hidden_dim']
        hidden_dim = hparams['term_hidden_dim']
        num_heads = hparams['term_heads']
        dropout = hparams['transformer_dropout']
        num_encoder_layers = hparams['term_layers']

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.input_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        edge_layer = TERMEdgeTransformerLayer if not hparams['term_use_mpnn'] else TERMEdgeMPNNLayer
        node_layer = TERMNodeTransformerLayer if not hparams['term_use_mpnn'] else TERMNodeMPNNLayer

        # Encoder layers
        self.edge_encoder = nn.ModuleList([
            edge_layer(hidden_dim, hidden_dim * 3 + (2 * hidden_dim if hparams['contact_idx'] else 0), dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.node_encoder = nn.ModuleList([
            node_layer(hidden_dim, num_in=hidden_dim * 2 + (2 * hidden_dim if hparams['contact_idx'] else 0), num_heads=num_heads, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V, E, E_idx, mask, contact_idx=None):
        """ Refine TERM graph representations

        Args
        ----
        V : torch.Tensor
            Node embeddings
            Shape: n_batches x n_terms x max_term_len x n_hidden
        E : torch.Tensor
            Edge embeddings in kNN dense form
            Shape: n_batches x n_terms x max_term_len x max_term_len x n_hidden
        E_idx : torch.LongTensor
            Edge indices
            Shape: n_batches x n_terms x max_term_len x max_term_len
        mask : torch.ByteTensor
            Mask for TERM resides
            Shape: n_batches x n_terms x max_term_len
        contact_idx : torch.Tensor
            Embedded contact indices
            Shape: n_batches x n_terms x max_term_len x n_hidden

        Returns
        -------
        h_V : torch.Tensor
            TERM node embeddings
        h_E : torch.Tensor
            TERM edge embeddings
        """
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_term_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for edge_layer, node_layer in zip(self.edge_encoder, self.node_encoder):
            h_EV_edges = cat_term_edge_endpoints(h_E, h_V, E_idx)
            if self.hparams['contact_idx']:
                h_EV_edges = cat_term_edge_endpoints(h_EV_edges, contact_idx, E_idx)
            h_E = edge_layer(h_E, h_EV_edges, E_idx, mask_E=mask_attend, mask_attend=mask_attend)

            if self.hparams['contact_idx']:
                h_EI = cat_term_edge_endpoints(h_E, contact_idx, E_idx)
            else:
                h_EI = h_E
            h_EV_nodes = cat_term_neighbors_nodes(h_V, h_EI, E_idx)
            h_V = node_layer(h_V, h_EV_nodes, mask_V=mask, mask_attend=mask_attend)

        h_E = self.W_out(h_E)
        h_E = merge_duplicate_term_edges(h_E, E_idx)

        return h_V, h_E
