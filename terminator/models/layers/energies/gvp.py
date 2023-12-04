import functools

import torch
from torch import nn
import torch.nn.functional as F

from ..gvp import (GVP, Dropout, GVPConvLayer, LayerNorm, _merge, _split, tuple_cat, tuple_index, tuple_sum)
from ..utils import merge_duplicate_edges_geometric, merge_duplicate_pairE_geometric

# pylint: disable=no-member


class EdgeLayer(nn.Module):
    """ GVP Edge MPNN """
    def __init__(self,
                 node_dims,
                 edge_dims,
                 drop_rate=0.1,
                 n_layers=3,
                 module_list=None,
                 activations=(F.relu, torch.sigmoid),
                 vector_gate=False):
        super().__init__()
        self.si, self.vi = node_dims
        self.so, self.vo = edge_dims
        self.se, self.ve = edge_dims

        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        # Edge Messages
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), edge_dims))
                for _ in range(n_layers - 2):
                    module_list.append(GVP_(edge_dims, edge_dims))
                module_list.append(GVP_(edge_dims, edge_dims, activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

        # norm and dropout
        self.norm = nn.ModuleList([LayerNorm(edge_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        # FFN
        n_feedforward = 2
        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(edge_dims, edge_dims, activations=(None, None)))
        else:
            hid_dims = 4 * edge_dims[0], 2 * edge_dims[1]
            ff_func.append(GVP_(edge_dims, hid_dims))
            for _ in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, edge_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, h_V, edge_index, h_E, node_mask=None):
        """ TODO """
        h_V_merge = _merge(*h_V)
        fake_h_dim = h_V_merge.shape[-1]
        edge_index_flat = edge_index.flatten()
        edge_index_flat = edge_index_flat.unsqueeze(-1).expand([-1, fake_h_dim])
        h_V_gather = torch.gather(h_V_merge, -2, edge_index_flat)
        h_V_gather = h_V_gather.view(list(edge_index.shape) + [fake_h_dim])

        h_V_ij = _split(h_V_gather, self.vi)
        h_V_i, h_V_j = zip(*map(torch.unbind, h_V_ij))

        h_EV = tuple_cat(h_V_i, h_E, h_V_j)
        dh = self.message_func(h_EV)

        # merge scalar features in edges
        dh_s = merge_duplicate_edges_geometric(dh[0], edge_index)
        dh = (dh_s, dh[1])

        x = h_E
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))

        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x


class GVPPairEnergies(nn.Module):
    '''GNN Potts Model Encoder using GVP

    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP layers
    :param num_layers: number of GVP layers in each of the encoder
                       and decoder modules
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, hparams):

        self.hparams = hparams
        node_in_dim = (hparams['energies_input_dim'] + 6, 3)
        node_h_dim = (100, 16)
        edge_in_dim = (hparams['energies_input_dim'] + 32, 1)
        edge_h_dim = (32, 1)
        num_layers = hparams['energies_encoder_layers']
        drop_rate = hparams['energies_dropout']
        output_dim = (hparams['energies_output_dim'], 0)

        super().__init__()

        self.W_v = nn.Sequential(GVP(node_in_dim, node_h_dim, activations=(None, None)), LayerNorm(node_h_dim))
        self.W_e = nn.Sequential(GVP(edge_in_dim, edge_h_dim, activations=(None, None)), LayerNorm(edge_h_dim))

        self.node_encoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate, activations=(F.relu, F.relu))
            for _ in range(num_layers))
        self.edge_encoder_layers = nn.ModuleList(
            EdgeLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate, activations=(F.relu, F.relu))
            for _ in range(num_layers))

        self.W_out = GVP(edge_h_dim, output_dim, activations=(None, None))

        # if enabled, generate self energies in etab from node embeddings
        if "node_self_sub" in hparams.keys() and hparams["node_self_sub"] is True:
            self.W_proj = GVP(node_h_dim, (20, 0), activations=(None, None))

    def forward(self, h_V, edge_index, h_E):
        '''Forward pass to be used at train-time, or evaluating likelihood.

        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        '''
        local_dev = self.W_v[0].wh.weight.device  # slightly hacky way of getting current device
        h_V = (h_V[0].to(local_dev), h_V[1].to(local_dev))
        h_E = (h_E[0].to(local_dev), h_E[1].to(local_dev))
        edge_index = edge_index.to(local_dev)

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for node_layer, edge_layer in zip(self.node_encoder_layers, self.edge_encoder_layers):
            h_V = node_layer(h_V, edge_index, h_E)
            h_E = edge_layer(h_V, edge_index, h_E)

        etab = self.W_out(h_E)
        # merge pairE tables to ensure each edge has the same energy dist
        etab = merge_duplicate_pairE_geometric(etab, edge_index)

        # zero out off-diag energies for self edges
        self_edge_select = (edge_index[0] == edge_index[1])
        etab[self_edge_select] = etab[self_edge_select] * torch.eye(20).view(1, -1).to(etab.device)

        # if specified, use generate self energies from node embeddings
        if "node_self_sub" in self.hparams.keys() and self.hparams["node_self_sub"] is True:
            h_V = self.W_proj(h_V)
            etab_select = etab[(edge_index[0] == edge_index[1])]
            etab[edge_index[0] == edge_index[1]] = torch.diag_embed(h_V, dim1=-2, dim2=-1).view(etab_select.shape)

        return etab, edge_index
