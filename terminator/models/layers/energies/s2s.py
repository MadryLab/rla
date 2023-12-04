""" GNN Potts Model Encoder modules

This file contains the GNN Potts Model Encoder, as well as an ablated version of
itself. """
from __future__ import print_function

import torch
from torch import nn

from terminator.models.layers.graph_features import MultiChainProteinFeatures
from terminator.models.layers.s2s_modules import (EdgeMPNNLayer, EdgeTransformerLayer, NodeMPNNLayer,
                                                  NodeTransformerLayer)
from terminator.models.layers.utils import (cat_edge_endpoints, cat_neighbors_nodes, gather_edges, gather_nodes,
                                            merge_duplicate_pairE, all_comb_to_per_node, all_comb_to_per_node_transpose, 
                                            per_node_to_all_comb, sync_inds_shape)
import time
from torch_scatter import scatter_mean
from torch.nn.utils.rnn import pad_sequence
from terminator.models.layers.transformer import GraphTransformer
# pylint: disable=no-member, not-callable

def merge_dups(h_E, inv_mapping):
    orig_shape = h_E.shape
    flattened = h_E.flatten(1, 2)
    condensed = scatter_mean(flattened, inv_mapping, dim=1)
    expanded_inv_mapping = inv_mapping.unsqueeze(-1).expand((-1, -1, orig_shape[-1]))
    rescattered = torch.gather(flattened, dim=1, index=expanded_inv_mapping)
    rescattered = rescattered.unflatten(1, (orig_shape[1], orig_shape[2]))
    return rescattered

def get_merge_dups_mask(E_idx):
    N = E_idx.shape[1]
    tens_place = torch.arange(N).cuda().unsqueeze(0).unsqueeze(-1)
    # tens_place = tens_place.unsqueeze(0).unsqueeze(-1)
    min_val = torch.minimum(E_idx, tens_place)
    max_val = torch.maximum(E_idx, tens_place)
    edge_indices = min_val*N + max_val
    edge_indices = edge_indices.flatten(1,2)
    unique_inv = []
    max_num_edges = 0
    for b in range(len(edge_indices)):
        uniq, inv = torch.unique(edge_indices[b], return_inverse=True)
        unique_inv.append(inv)
        max_num_edges = max(0, len(uniq))
    unique_inv = torch.stack(unique_inv)
    return unique_inv, max_num_edges

class AblatedPairEnergies(nn.Module):
    """Ablated GNN Potts Model Encoder

    Attributes
    ----------
    dev: str
        Device representing where the model is held
    hparams: dict
        Dictionary of parameter settings (see :code:`terminator/utils/model/default_hparams.py`)
    features : MultiChainProteinFeatures
        Module that featurizes a protein backbone (including multimeric proteins)
    W : nn.Linear
        Output layer that projects edge embeddings to proper output dimensionality
    """
    def __init__(self, hparams):
        """ Graph labeling network """
        super().__init__()
        hdim = hparams['energies_hidden_dim']
        self.hparams = hparams

        # Featurization layers
        self.features = MultiChainProteinFeatures(node_features=hdim,
                                                  edge_features=hdim,
                                                  top_k=hparams['k_neighbors'],
                                                  features_type=hparams['energies_protein_features'],
                                                  augment_eps=hparams['energies_augment_eps'],
                                                  dropout=hparams['energies_dropout'],
                                                  num_positional_embeddings=hparams['num_positional_embeddings'],
                                                  zero_out_pos_embs=hparams['zero_out_pos_embs'],
                                                 )

        self.W = nn.Linear(hparams['energies_input_dim'] * 3, hparams['energies_output_dim'])

    def forward(self, V_embed, E_embed, X, x_mask, chain_idx):
        """ Create kNN etab from TERM features, then project to proper output dimensionality.

        Args
        ----
        V_embed : torch.Tensor
            TERM node embeddings
            Shape: n_batch x n_res x n_hidden
        E_embed : torch.Tensor
            TERM edge embeddings
            Shape : n_batch x n_res x n_res x n_hidden
        X : torch.Tensor
            Backbone coordinates
            Shape: n_batch x n_res x 4 x 3
        x_mask : torch.ByteTensor
            Mask for X.
            Shape: n_batch x n_res
        chain_idx : torch.LongTensor
            Indices such that each chain is assigned a unique integer and each residue in that chain
            is assigned that integer.
            Shape: n_batch x n_res

        Returns
        -------
        etab : torch.Tensor
            Energy table in kNN dense form
            Shape: n_batch x n_res x k x n_hidden
        E_idx : torch.LongTensor
            Edge index for `etab`
            Shape: n_batch x n_res x k
        """
        # compute the kNN etab
        _, _, E_idx = self.features(X, chain_idx, x_mask)  # notably, we throw away the backbone features
        E_embed_neighbors = gather_edges(E_embed, E_idx)
        h_E = cat_edge_endpoints(E_embed_neighbors, V_embed, E_idx)
        etab = self.W(h_E)

        # merge duplicate pairEs
        n_batch, n_res, k, out_dim = etab.shape
        # ensure output etab is masked properly
        etab = etab * x_mask.view(n_batch, n_res, 1, 1)
        etab = etab.unsqueeze(-1).view(n_batch, n_res, k, 20, 20)
        etab[:, :, 0] = etab[:, :, 0] * torch.eye(20).to(etab.device) # zero off-diagonal energies
        etab = merge_duplicate_pairE(etab, E_idx)
        etab = etab.view(n_batch, n_res, k, out_dim)

        return etab, E_idx


class Featurizer(nn.Module):
    """Perform featurization"""
    def __init__(self, hparams):
        """ Graph labeling network """
        super().__init__()
        self.hparams = hparams
        hdim = hparams['energies_hidden_dim']

        # Hyperparameters
        output_dim = hparams['energies_output_dim']
        dropout = hparams['energies_dropout']
        num_encoder_layers = hparams['energies_encoder_layers']

        # Featurization layers
        self.features = MultiChainProteinFeatures(node_features=hdim,
                                                  edge_features=hdim,
                                                  top_k=hparams['k_neighbors'],
                                                  features_type=hparams['energies_protein_features'],
                                                  features_options=hparams['condense_options'],
                                                  augment_eps=hparams['energies_augment_eps'],
                                                  dropout=hparams['energies_dropout'],
                                                  chain_handle=hparams['chain_handle'],
                                                  num_positional_embeddings=hparams['num_positional_embeddings'],
                                                  zero_out_pos_embs=hparams['zero_out_pos_embs'],
                                                 )

        # Embedding layers
        self.W_v = nn.Linear(hdim + hparams['energies_input_dim'], hdim, bias=True)
        self.W_e = nn.Linear(hdim + hparams['energies_input_dim'], hdim, bias=True)
        
    def forward(self, V_embed, E_embed, X, x_mask, chain_idx):
        # Prepare node and edge embeddings
        if self.hparams['energies_input_dim'] != 0:
            V, E, E_idx = self.features(X, chain_idx, x_mask)
            if not self.hparams['use_coords']:  # this is hacky/inefficient but i am lazy
                V = torch.zeros_like(V)
                E = torch.zeros_like(E)
            # fuse backbone and TERM embeddings
            V = torch.cat([V, V_embed], dim=-1)
            E_embed_neighbors = gather_edges(E_embed, E_idx)
            E = torch.cat([E, E_embed_neighbors], dim=-1)
            h_V = self.W_v(V)
            h_E = self.W_e(E)
        elif V_embed is not None and E_embed is None:
            _, E, E_idx = self.features(X, chain_idx, x_mask, need_node_embeddings=False)
            h_V = V_embed
            h_E = self.W_e(E)
        else:
            # just use backbone features
            V, E, E_idx = self.features(X, chain_idx, x_mask)
            h_V = self.W_v(V)
            h_E = self.W_e(E)
        return h_V, h_E, E_idx

class PairEnergies(nn.Module):
    """GNN Potts Model Encoder
    Attributes
    ----------
    dev: str
        Device representing where the model is held
    hparams: dict
        Dictionary of parameter settings (see :code:`terminator/utils/model/default_hparams.py`)
    features : MultiChainProteinFeatures
        Module that featurizes a protein backbone (including multimeric proteins)
    W_v : nn.Linear
        Embedding layer for incoming TERM node embeddings
    W_e : nn.Linear
        Embedding layer for incoming TERM edge embeddings
    edge_encoder : nn.ModuleList of EdgeTransformerLayer or EdgeMPNNLayer
        Edge graph update layers
    node_encoder : nn.ModuleList of NodeTransformerLayer or NodeMPNNLayer
        Node graph update layers
    W_out : nn.Linear
        Output layer that projects edge embeddings to proper output dimensionality
    W_proj : nn.Linear (optional)
        Output layer that projects node embeddings to proper output dimensionality.
        Enabled when :code:`hparams["node_self_sub"]=True`
    """
    def __init__(self, hparams):
        """ Graph labeling network """
        super().__init__()

        self.hparams = hparams
        self.featurizer = Featurizer(hparams)
        hdim = hparams['energies_hidden_dim']
        self.clip_mode = hparams.get("clip_mode", False)

        # Hyperparameters
        output_dim = hparams['energies_output_dim']
        dropout = hparams['energies_dropout']
        num_encoder_layers = hparams['energies_encoder_layers']

        # Embedding layers
        merge_edges = hparams['condense_options'].find('reduce_edges') == -1
        is_mpnn = hparams['energies_type'] = 'MPNN'
        if hparams['energies_type'] == 'MPNN':
            self.edge_encoder = nn.ModuleList([
                EdgeMPNNLayer(hdim, hdim * 3, merge_edges=merge_edges, dropout=dropout)
                for _ in range(num_encoder_layers)
            ])
            self.node_encoder = nn.ModuleList([
                NodeMPNNLayer(hdim, hdim * 2, dropout=dropout) for _ in range(num_encoder_layers)
            ])
        else:
            self.edge_encoder = nn.ModuleList([
                EdgeTransformerLayer(hdim, hdim * 3, dropout=dropout)
                for _ in range(num_encoder_layers)
            ])
            self.node_encoder = nn.ModuleList([
                NodeTransformerLayer(hdim, hdim * 2, dropout=dropout) for _ in range(num_encoder_layers)
            ])

        # if enabled, generate self energies in etab from node embeddings
        if "node_self_sub" in hparams.keys() and hparams["node_self_sub"] is True:
            self.W_proj = nn.Linear(hidden_dim, 20)

        # project edges to proper output dimensionality
        if not self.clip_mode:
            self.W_out = nn.Linear(hidden_dim, output_dim, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        self.merge_dup_fn = merge_dups
       

    def forward(self, V_embed, E_embed, X, x_mask, chain_idx):
        """ Create kNN etab from backbone and TERM features, then project to proper output dimensionality.
        Args
        ----
        V_embed : torch.Tensor or None
            TERM node embeddings. None only accepted if :code:`hparams['energies_input_dim']=0`.
            Shape: n_batch x n_res x n_hidden
        E_embed : torch.Tensor or None
            TERM edge embeddings. None only accepted if :code:`hparams['energies_input_dim']=0`.
            Shape : n_batch x n_res x n_res x n_hidden
        X : torch.Tensor
            Backbone coordinates
            Shape: n_batch x n_res x 4 x 3
        x_mask : torch.ByteTensor
            Mask for X.
            Shape: n_batch x n_res
        chain_idx : torch.LongTensor
            Indices such that each chain is assigned a unique integer and each residue in that chain
            is assigned that integer.
            Shape: n_batch x n_res
        inds_convert : tuple of torch.Tensor
                Indexes needed to convert from expanded (directed) to reduced (undirected) dimensionalities
        mask_reduced : torch.ByteTensor
            Mask in reduced dimensionality
        Returns
        -------
        etab : torch.Tensor
            Energy table in kNN dense form
            Shape: n_batch x n_res x k x n_hidden
        h_V : torch.Tensor
            Node representation
            Shape: n_batch x n_res x n_hidden
        E_idx : torch.LongTensor
            Edge index for `etab`
            Shape: n_batch x n_res x k
        """
        # Prepare node and edge embeddings
        
        h_V, h_E, E_idx = self.featurizer(V_embed, E_embed, X, x_mask, chain_idx)

        # Graph updates
        mask_attend = gather_nodes(x_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = x_mask.unsqueeze(-1) * mask_attend

        inv_mapping, max_edges = get_merge_dups_mask(E_idx)
        h_E = self.merge_dup_fn(h_E, inv_mapping)
        

        for edge_layer, node_layer in zip(self.edge_encoder, self.node_encoder):
            h_EV_edges = cat_edge_endpoints(h_E, h_V, E_idx)
            h_E = edge_layer(h_E, h_EV_edges, E_idx, mask_E=x_mask, mask_attend=mask_attend)
            h_E = self.merge_dup_fn(h_E, inv_mapping)
            h_EV_nodes = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = node_layer(h_V, h_EV_nodes, mask_V=x_mask, mask_attend=mask_attend)

        if self.clip_mode:
            # break out early, rest isn't supervised
            return h_E, h_V, E_idx

        h_E = self.W_out(h_E)
        n_batch, n_res, k, out_dim = h_E.shape
        h_E = h_E * x_mask.view(n_batch, n_res, 1, 1) # ensure output etab is masked properly
        h_E = h_E.unsqueeze(-1).view(n_batch, n_res, k, 20, 20)
        h_E[:, :, 0] = h_E[:, :, 0] * torch.eye(20).to(h_E.device) # zero off-diagonal energies
        h_E = merge_duplicate_pairE(h_E, E_idx)
        # if specified, use generate self energies from node embeddings
        if "node_self_sub" in self.hparams.keys() and self.hparams["node_self_sub"] is True:
            h_V = self.W_proj(h_V)
            h_E[..., 0, :, :] = torch.diag_embed(h_V, dim1=-2, dim2=-1)

        # reshape to fit kNN output format
        h_E = h_E.view(n_batch, n_res, k, out_dim)
        #h_V = torch.reshape(h_V, (h_V.shape[0], h_V.shape[1]*h_V.shape[2]))
        return h_E, h_V, E_idx

class TransformerPairEnergies(nn.Module):
    """TransformerPairEnergies
    """
    def __init__(self, hparams):
        """ Graph labeling network """
        super().__init__()
        self.hparams = hparams
        self.featurizer = Featurizer(hparams)
        hdim = hparams['energies_hidden_dim']
                
        graphformer_config = hparams['graphformer_config']
        dropout = 0
        self.transformer = GraphTransformer(
            num_in=hdim, num_e_in=hdim, num_heads=graphformer_config['num_heads'],
            num_layers=graphformer_config['num_layers'], embed_per_head=graphformer_config['embed_per_head'], 
            dropout = dropout, mlp_multiplier=graphformer_config['mlp_multiplier'], 
            num_out=hdim )
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V_embed, E_embed, X, x_mask, chain_idx):
        """ Create kNN etab from backbone and TERM features, then project to proper output dimensionality.

        Args
        ----
        V_embed : torch.Tensor or None
            TERM node embeddings. None only accepted if :code:`hparams['energies_input_dim']=0`.
            Shape: n_batch x n_res x n_hidden
        E_embed : torch.Tensor or None
            TERM edge embeddings. None only accepted if :code:`hparams['energies_input_dim']=0`.
            Shape : n_batch x n_res x n_res x n_hidden
        X : torch.Tensor
            Backbone coordinates
            Shape: n_batch x n_res x 4 x 3
        x_mask : torch.ByteTensor
            Mask for X.
            Shape: n_batch x n_res
        chain_idx : torch.LongTensor
            Indices such that each chain is assigned a unique integer and each residue in that chain
            is assigned that integer.
            Shape: n_batch x n_res
        """
        # Prepare node and edge embeddings
        h_V, h_E, E_idx = self.featurizer(V_embed, E_embed, X, x_mask, chain_idx)
        # Graph updates
        mask_attend = gather_nodes(x_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = x_mask.unsqueeze(-1) * mask_attend
        h_V = self.transformer(x=h_V, E_idx=E_idx, E_features=h_E, e_mask=mask_attend, x_mask=x_mask)
        return h_E, h_V, E_idx