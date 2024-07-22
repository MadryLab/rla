
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from terminator.data import noise as noise
from terminator.models.layers.s2s_modules import EdgeMPNNLayer, NodeMPNNLayer, NodeTransformerLayer, NeighborAttention
from terminator.models.layers.utils import extract_knn, cat_edge_endpoints, cat_neighbors_nodes, gather_edges, gather_nodes, merge_duplicate_pairE
from clip_main import get_wds_loaders
import src.data_utils as data_utils
from rla_utils import get_inter_dists, get_interaction_res
sys.path.insert(0, '/data1/groups/keatinglab/tools')
sys.path.insert(0, '/home/gridsan/fbirnbaum/TERMinator/scripts/design_sequence')
import argparse
import json
import copy
import pickle

HOLD_OUTS = {
    'IL7Ra': 'bcov_4h_v1',
    'SARS_CoV2_RBD': 'HHH_b1',
    'FGFR2': 'bcov_4h_v1',
    'InsulinR': 'bcov_4h_v1',
    'PDGFR': 'lx_HEEHE', #   bcov_4h_v1
}

class WDS_Args():
    def __init__(self, train_wds_path, val_wds_path, data_root='/data1/groups/keating_madry/wds'):
        self.data_root = data_root
        self.train_wds_path = train_wds_path
        self.val_wds_path = val_wds_path
        self.zip_enabled = False
        self.distributed = False
        self.world_size = 1
        self.dist_val_len = 4200
        self.max_coord_len = 2000
        self.max_seq_len = 1024
        self.burn_in = 0
        self.num_mutations = False
        self.masked_rate = -1
        self.masked_mode = 'MASK'
        self.batch_size = 1
        self.num_workers = 10

def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BinderModel(nn.Module):
    def __init__(self, in_features, out_features, num_layers, predict_prob = True, activation_layers='relu', bias=True, dropout=0):
        super(BinderModel, self).__init__()
        self.activation_layers = activation_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_features[i], out_features[i], bias=bias))
        self.dropout_prob = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout, inplace=False)
        self.predict_prob = predict_prob
        if predict_prob:
            self.prob_activation = torch.nn.Sigmoid()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            
    def forward(self, x, mask, coords=None):
        for i, layer in enumerate(self.layers):  
            # print('\t', x.shape)
            x = layer(x)
            if (i == len(self.layers) - 1) and self.predict_prob:
                # x *= mask
                continue
            if self.activation_layers == 'relu':
                x = F.relu(x)
            else:
                x = gelu(x)
            # x *= mask
        if self.dropout_prob > 0:
            x = self.dropout(x)
        if self.predict_prob:
            x = self.prob_activation(x).squeeze(-1)
        else:
            x = x.squeeze(-1)
        return x
    

def merge_dups(h_E, inv_mapping):
    orig_shape = h_E.shape
    flattened = h_E.flatten(1, 2)
    # condensed = scatter_mean(flattened, inv_mapping, dim=1)
    expanded_inv_mapping = inv_mapping.unsqueeze(-1).expand((-1, -1, orig_shape[-1]))
    rescattered = torch.gather(flattened, dim=1, index=expanded_inv_mapping)
    rescattered = rescattered.unflatten(1, (orig_shape[1], orig_shape[2]))
    return rescattered

def get_merge_dups_mask(E_idx):
    N = E_idx.shape[1]
    tens_place = torch.arange(N).unsqueeze(0).unsqueeze(-1)
    if E_idx.device.type != 'cpu':
        tens_place = tens_place.cuda()
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

def _dist(X, mask, eps=1E-6, top_k=30):
    # Convolutional network on NCHW
    mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
    D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

    # Identify k nearest neighbors (including self)
    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - mask_2D) * D_max
    top_k = min(top_k, D.shape[1])
    D_neighbors, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)

    return D_neighbors, E_idx

class BinderGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, node_hidden_dim, edge_hidden_dim, num_hidden_layers, dropout, predict_prob=True, use_dummy=True):
        super(BinderGNN, self).__init__()
        # self.W_V = nn.Linear(node_in_dim, node_hidden_dim)
        # self.W_V_nl = gelu
        self.embedding = nn.Embedding(102, node_hidden_dim, padding_idx=0)
        self.W_E = nn.Linear(edge_in_dim, edge_hidden_dim)
        self.W_E_nl = gelu

        self.node_encoder = nn.ModuleList(
            [NodeMPNNLayer(node_hidden_dim, node_hidden_dim * 2, dropout=dropout) for _ in range(num_hidden_layers)]
        )
        self.edge_encoder = nn.ModuleList(
            [EdgeMPNNLayer(edge_hidden_dim, edge_hidden_dim * 3, dropout=dropout) for _ in range(num_hidden_layers)]
        )

        self.T_node_encoder = nn.ModuleList(
            [NodeTransformerLayer(node_hidden_dim, node_hidden_dim * 2, dropout=dropout) for _ in range(num_hidden_layers)]
        )

        self.W_out = nn.Linear(node_hidden_dim, 1)
        if predict_prob:
            self.W_act = torch.nn.Sigmoid()
        self.predict_prob = predict_prob
        self.use_dummy = use_dummy

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, rla, X, mask):
        rla = rla.to(X.device)
        mask = mask.to(X.device)
        E, E_idx = _dist(X, mask, eps=1E-6, top_k=30)
        # rla = torch.cat([rla, torch.mean(rla).unsqueeze(0)])
        # h_V = self.W_V_nl(self.W_V(rla.unsqueeze(-1)))
        # V = self.W_V(rla.unsqueeze(-1))
        V = self.embedding(rla)
        # rla_mean = torch.sum(rla, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        # rla_mean = torch.zeros_like(rla_mean)
        rla_mean = torch.div(torch.sum(rla, dim=1, keepdim=True), torch.sum(mask, dim=1, keepdim=True), rounding_mode='trunc').long()
        # h_T = self.W_V_nl(self.W_V(rla_mean.unsqueeze(-1)))
        h_T = self.embedding(rla_mean)
        h_E = self.W_E_nl(self.W_E(E.unsqueeze(-1)))
        # h_E = torch.zeros(h_V.shape[0], h_V.shape[1], E_idx.shape[2], h_V.shape[2]).to(device = h_V.device, dtype=h_V.dtype)
        # Graph updates
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        # T_mask_attend = torch.zeros(x_mask.shape[0], x_mask.shape[1], x_mask.shape[1]).to(device=mask_attend.device, dtype=mask_attend.dtype)
        inv_mapping, max_edges = get_merge_dups_mask(E_idx)
        h_E = merge_dups(h_E, inv_mapping)
        Tmask = torch.ones(mask.shape[0], 1).to(dtype=mask.dtype, device=mask.device)
        Tmask_attend = torch.unsqueeze(mask, dim=1)
        for i_layer, (edge_layer, node_layer, T_node_layer) in enumerate(zip(self.edge_encoder, self.node_encoder, self.T_node_encoder)):
            # print('=====')
            # print(V[:,0])
            # print(h_T)
            # print('-----')
            h_EV_edges = cat_edge_endpoints(h_E, V, E_idx)
            h_E = edge_layer(h_E, h_EV_edges, E_idx, mask_E=mask, mask_attend=mask_attend)
            h_E = merge_dups(h_E, inv_mapping)
            h_EV_nodes = cat_neighbors_nodes(V, h_E, E_idx)
            V = node_layer(V, h_EV_nodes, mask_V=mask, mask_attend=mask_attend)
            V_expand = V.unsqueeze(dim=1)
            h_T_expand = h_T.unsqueeze(2).expand(V_expand.shape)
            h_T = T_node_layer(h_T, torch.cat([h_T_expand, V_expand], dim=-1), mask_V=Tmask, mask_attend=Tmask_attend, update=(i_layer > 0))
            # print('!!!!!')
            # print(V[:,0])
            # print(h_T)
            # print('??????')
        
        if not self.use_dummy:
            h_T = torch.mean(V, dim=1, keepdim=True)
        h_T = self.W_out(h_T).squeeze(-1).squeeze(-1)
        if self.predict_prob:
            h_T = self.W_act(h_T)
            
        return h_T
    
class BinderTransformerModel(nn.Transformer):
    def __init__(self, d_in=1, d_model=32, nhead=1, num_encoder_layers=2, dim_feedforward=32, dropout=0.0, predict_prob=True, batch_first=True, use_dummy=True):
        super(BinderTransformerModel, self).__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_encoder_layers=num_encoder_layers, dropout=dropout, batch_first=True)
        # self.h_V = nn.Linear(d_in, d_model)
        # self.h_V_nl = gelu

        self.decoder = nn.Linear(d_model, 1)
        self.decoder_nl = gelu
        if predict_prob:
            self.act = torch.nn.Sigmoid()
        self.use_dummy = use_dummy
        self.nhead = nhead

        self.embedding = nn.Embedding(102, d_model, padding_idx=0)

        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.normal_(p)

    def forward(self, src, X, mask):
        src = src.to(device=X.device)
        mask = mask.to(device=X.device)
        _, E_idx = _dist(X, mask, eps=1E-6, top_k=30)
        src_mean = torch.div(torch.sum(src, dim=1, keepdim=True), torch.sum(mask, dim=1, keepdim=True), rounding_mode='trunc').long()
        # src_mean = torch.zeros_like(src_mean)
        # V = self.h_V(src.unsqueeze(-1))
        V = self.embedding(src).to(dtype=torch.float64)
        # h_V = src.unsqueeze(-1)
        V_mean = self.embedding(src_mean).to(dtype=torch.float64)
        # h_V = torch.cat([h_V, src_mean.unsqueeze(-1)], dim=1)
        V = torch.cat([V, V_mean], dim=1)
        base_mask = torch.ones(X.shape[0], X.shape[1], X.shape[1]).to(device=X.device, dtype=X.dtype)
        base_src = torch.zeros(X.shape[0], X.shape[1], X.shape[1]).to(device=X.device, dtype=X.dtype)
        attend_mask = base_mask.scatter(2, E_idx, base_src)
        attend_mask = F.pad(attend_mask, (0, 1), value=1)
        attend_mask = F.pad(attend_mask, (0, 0, 0, 1), value=0)
        attend_mask[:,-1,:-1] += (1 - mask)
        pad_mask = F.pad(mask, (0, 1), value=1)
        # attend_mask = 1 - attend_mask
        attend_mask[attend_mask == 1] = float('-inf')
        attend_mask = attend_mask.repeat(self.nhead, 1, 1)
        
        # attend_mask = attend_mask.to(torch.bool)
        # print(attend_mask.shape)
        print(V)
        output = self.encoder(V) #mask=attend_mask
        print(output)
        output = self.decoder(output[:,-1]).squeeze(-1)
        output = self.act(output)
        return output    
    

def package_data_old(df_ids, pep_data, prot_data, base_data, rla_types, max_chain_len):
    data_tensor = []
    for id_ in df_ids:
        pep_len = len(pep_data[id_])
        prot_len = len(prot_data[id_])
        complex_tensor = torch.Tensor(base_data[id_])
        pep_complex_tensor = complex_tensor[:pep_len]
        prot_complex_tensor = complex_tensor[pep_len:]
        pep_tensor = torch.Tensor(pep_data[id_])
        prot_tensor = torch.Tensor(prot_data[id_])
        if 'protein' in rla_types and 'peptide' in rla_types:
            pep_weave = F.pad(torch.stack((pep_complex_tensor, pep_tensor)).view(pep_len*2), (0, max_chain_len - 2*pep_len), mode='constant', value=0)
            prot_weave = F.pad(torch.stack((prot_complex_tensor, prot_tensor)).view(prot_len*2), (0, max_chain_len - 2*prot_len), mode='constant', value=0)
            final_tensor = torch.cat([pep_weave, prot_weave])
        elif 'pep_cat' in rla_types:
            pep_cat = torch.cat([pep_complex_tensor, pep_tensor, prot_complex_tensor])
            final_tensor = F.pad(pep_cat, (0, MAX_LEN - pep_cat.shape[0]), mode='constant', value=0)
        elif 'only_pep' in rla_types:
            final_tensor = F.pad(pep_tensor, (0, MAX_LEN - pep_tensor.shape[0]), mode='constant', value=0)
        elif 'pep_interweave' in rla_types:
            pep_weave = F.pad(torch.stack((pep_complex_tensor, pep_tensor)).view(pep_len*2), (0, max_chain_len - 2*pep_len), mode='constant', value=0)
            prot_final = F.pad(prot_complex_tensor, (0, max_chain_len - prot_len), mode='constant', value=0)
            final_tensor = torch.cat([pep_weave, prot_final])
        else:
            final_tensor = torch.cat([F.pad(pep_complex_tensor, (0, max_chain_len - pep_len), mode='constant', value=0), F.pad(prot_complex_tensor, (0, max_chain_len - prot_len), mode='constant', value=0)])
        data_tensor.append(final_tensor)
    
    data_tensor = torch.stack(data_tensor, dim=0)
    return data_tensor

def run_iter_old(base_data, pep_data, prot_data, df, binder_model, optimizer, loss_fn, task, rla_types, dev, grad, coord_data=None, batch_size=-1):
    print('using old run_iter!')
    if grad:
        torch.set_grad_enabled(True)
        binder_model.train()
    else:
        torch.set_grad_enabled(False)
        binder_model.eval()

    # Subsample df and dataset to get balanced batch
    df_pos_ids = df[df['binder_4000_nm']]['description'].values
    df_all_neg_ids = df[~df['binder_4000_nm']]['description'].values
    max_chain_len = MAX_LEN // 2
    for _ in range(100):
        np.random.shuffle(df_all_neg_ids)
        df_neg_ids = df_all_neg_ids[:len(df_pos_ids)]

        pos_tensor = package_data_old(df_pos_ids, pep_data, prot_data, base_data, rla_types, max_chain_len)
        pos_labels = torch.ones(pos_tensor.shape[0])
        
        neg_tensor = package_data_old(df_neg_ids, pep_data, prot_data, base_data, rla_types, max_chain_len)
        neg_labels = torch.zeros(neg_tensor.shape[0])
        batch_tensor = torch.cat((pos_tensor, neg_tensor), dim=0)
        indices = torch.randperm(batch_tensor.shape[0])
        batch_tensor = batch_tensor[indices]
        batch_labels = torch.cat((pos_labels, neg_labels), dim=0)
        batch_labels = batch_labels[indices].to(device=dev)
        # Pass batch residue-level RLA scores through binder_model to get probabilities
        binder_model = binder_model.to(dtype=torch.float64)
        num_batches = math.ceil(batch_tensor.shape[0] / batch_size)
        for i in range(num_batches):
            cur_batch_tensor = batch_tensor[i*batch_size:(i+1)*batch_size].to(dtype=torch.float64)
            cur_batch_labels = batch_labels[i*batch_size:(i+1)*batch_size].to(dtype=torch.float64)
            batch_probs = binder_model(cur_batch_tensor.to(device=dev, dtype=torch.float64), None)

            # Apply F.cross_entropy to get loss
            loss = loss_fn(batch_probs, cur_batch_labels)
            if grad:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return loss.item()

def package_data(df_ids, pep_data, prot_data, base_data, coord_data, rla_types, max_chain_len):
    MAX_LEN = max_chain_len * 2
    data_tensor = []
    data_mask = []
    for id_ in df_ids:
        mask = torch.zeros(MAX_LEN)
        pep_len = len(pep_data[id_])
        prot_len = len(prot_data[id_])
        pep_inter_res = coord_data[id_][0]
        pep_dist_info = coord_data[id_][1]
        # pep_dist_info = dist_info[:pep_len]
        sorted_pep_dist = np.argsort(pep_dist_info)
        pep_dist_rank = np.empty_like(sorted_pep_dist)
        pep_dist_rank[sorted_pep_dist] = torch.from_numpy(np.arange(len(pep_dist_info)))
        prot_inter_res = coord_data[id_][2]
        prot_dist_info = coord_data[id_][3]
        # prot_dist_info = dist_info[pep_len:]
        sorted_prot_dist = np.argsort(prot_dist_info)
        prot_dist_rank = np.empty_like(sorted_prot_dist)
        prot_dist_rank[sorted_prot_dist] = torch.from_numpy(np.arange(len(prot_dist_info)))
        
        complex_tensor = torch.Tensor(base_data[id_])
        pep_complex_tensor = complex_tensor[:pep_len][pep_inter_res].contiguous()[pep_dist_rank]
        prot_complex_tensor = complex_tensor[pep_len:][prot_inter_res].contiguous()[prot_dist_rank]
        pep_tensor = torch.Tensor(pep_data[id_])[pep_inter_res].contiguous()[pep_dist_rank]
        prot_tensor = torch.Tensor(prot_data[id_])[prot_inter_res].contiguous()[prot_dist_rank]
        if 'protein' in rla_types and 'peptide' in rla_types:
            pep_weave = F.pad(torch.stack((pep_complex_tensor, pep_tensor)).view(pep_tensor.shape[0]*2), (0, max_chain_len - 2*pep_len), mode='constant', value=0)
            prot_weave = F.pad(torch.stack((prot_complex_tensor, prot_tensor)).view(prot_tensor.shape[0]*2), (0, max_chain_len - 2*prot_len), mode='constant', value=0)
            final_tensor = torch.cat([pep_weave, prot_weave])
        elif 'pep_cat' in rla_types:
            pep_cat = torch.cat([pep_complex_tensor, pep_tensor, prot_complex_tensor])
            final_tensor = F.pad(pep_cat, (0, MAX_LEN - pep_cat.shape[0]), mode='constant', value=0)
        elif 'only_pep' in rla_types:
            final_tensor = F.pad(pep_tensor, (0, MAX_LEN - pep_tensor.shape[0]), mode='constant', value=0)
        elif 'pep_interweave' in rla_types:
            pep_weave = F.pad(torch.stack((pep_complex_tensor, pep_tensor)).view(pep_tensor.shape[0]*2), (0, max_chain_len - 2*pep_len), mode='constant', value=0)
            prot_final = F.pad(prot_complex_tensor, (0, max_chain_len - prot_complex_tensor.shape[0]), mode='constant', value=0)
            final_tensor = torch.cat([pep_weave, prot_final])
        else:
            final_tensor = torch.cat([F.pad(pep_complex_tensor, (0, max_chain_len - pep_complex_tensor.shape[0]), mode='constant', value=0), F.pad(prot_complex_tensor, (0, max_chain_len - prot_complex_tensor.shape[0]), mode='constant', value=0)])
        data_tensor.append(final_tensor)
        mask[final_tensor != 0] = 1
        data_mask.append(mask)
    data_tensor = torch.stack(data_tensor, dim=0)
    data_mask = torch.stack(data_mask, dim=0)

    return data_tensor, data_mask

def run_iter(base_data, pep_data, prot_data, df, binder_model, optimizer, loss_fn, task, rla_types, dev, grad, coord_data, batch_size=-1):
    if grad:
        torch.set_grad_enabled(True)
        binder_model.train()
    else:
        torch.set_grad_enabled(False)
        binder_model.eval()

    # Subsample df and dataset to get balanced batch
    df_pos_ids = df[df['binder_4000_nm']]['description'].values
    df_all_neg_ids = df[~df['binder_4000_nm']]['description'].values
    max_chain_len = MAX_LEN // 2
    df = df.replace(np.inf, np.nan)
    df = df.replace(np.nan, np.nanmax(df['kd_lb'].values))
    for _ in range(100):
        np.random.shuffle(df_all_neg_ids)
        df_neg_ids = df_all_neg_ids[:len(df_pos_ids)]
        

        pos_tensor, pos_mask = package_data(df_pos_ids, pep_data, prot_data, base_data, coord_data, rla_types, max_chain_len)
        if task == 'binary':
            pos_labels = torch.ones(pos_tensor.shape[0])
        elif task == 'regression':
            pos_labels = torch.from_numpy(df[df['binder_4000_nm']]['kd_lb'].values).to(torch.float64)
        
        neg_tensor, neg_mask = package_data(df_neg_ids, pep_data, prot_data, base_data, coord_data, rla_types, max_chain_len)
        if task == 'binary':
            neg_labels = torch.zeros(neg_tensor.shape[0])
        elif task == 'regression':
            neg_labels = torch.from_numpy(df[df['description'].isin(df_neg_ids)]['kd_lb'].values).to(torch.float64)
        batch_tensor = torch.cat((pos_tensor, neg_tensor), dim=0)
        batch_mask = torch.cat((pos_mask, neg_mask), dim=0)
        indices = torch.randperm(batch_tensor.shape[0])
        
        batch_tensor = batch_tensor[indices]
        batch_labels = torch.cat((pos_labels, neg_labels), dim=0)
        batch_labels = batch_labels[indices].to(device=dev, dtype=torch.float64)
        # Pass batch residue-level RLA scores through binder_model to get probabilities
        binder_model = binder_model.to(dtype=torch.float64)
        batch_mask = batch_mask[indices]
        # Pass batch residue-level RLA scores through binder_model to get probabilities
        batch_probs = binder_model(batch_tensor.to(device=dev, dtype=torch.float64), mask=batch_mask.to(device=dev))

        # Apply F.cross_entropy to get loss
        loss = loss_fn(batch_probs, batch_labels)
        if grad:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # num_batches = math.ceil(batch_tensor.shape[0] / batch_size)
            # for i in range(num_batches):
            #     cur_batch_tensor = batch_tensor[i*batch_size:(i+1)*batch_size]
            #     cur_batch_labels = batch_labels[i*batch_size:(i+1)*batch_size]
            #     cur_batch_mask = batch_mask[i*batch_size:(i+1)*batch_size]
            #     batch_probs = binder_model(cur_batch_tensor.to(device=dev, dtype=torch.float64), cur_batch_mask.to(device=dev))

            #     # Apply F.cross_entropy to get loss
            #     loss = loss_fn(batch_probs, cur_batch_labels)
            #     if grad:
            #         optimizer.zero_grad()
            #         loss.backward()
            #         optimizer.step()
    return loss.item()


def package_data_gnn(df_ids, pep_data, prot_data, base_data, rla_types, max_chain_len, bin_size=0.01):
    data_tensor = []
    data_coords = []
    data_mask = []
    for id_ in df_ids:
        mask = torch.zeros(MAX_LEN)
        pep_len = len(pep_data[id_])
        prot_len = len(prot_data[id_])
        complex_tensor = torch.Tensor(base_data[id_])
        complex_tensor = complex_tensor - torch.min(complex_tensor)
        complex_tensor = complex_tensor / torch.max(complex_tensor)
        complex_tensor = torch.div(complex_tensor, bin_size, rounding_mode='floor').long() + 1
        pep_complex_tensor = complex_tensor[:pep_len]
        prot_complex_tensor = complex_tensor[pep_len:]
        pep_tensor = torch.Tensor(pep_data[id_])
        prot_tensor = torch.Tensor(prot_data[id_])
        if 'protein' in rla_types and 'peptide' in rla_types:
            pep_weave = F.pad(torch.stack((pep_complex_tensor, pep_tensor)).view(pep_len*2), (0, max_chain_len - 2*pep_len), mode='constant', value=0)
            prot_weave = F.pad(torch.stack((prot_complex_tensor, prot_tensor)).view(prot_len*2), (0, max_chain_len - 2*prot_len), mode='constant', value=0)
            final_tensor = torch.cat([pep_weave, prot_weave])
        elif 'pep_cat' in rla_types:
            pep_cat = torch.cat([pep_complex_tensor, pep_tensor, prot_complex_tensor])
            final_tensor = F.pad(pep_cat, (0, MAX_LEN - pep_cat.shape[0]), mode='constant', value=0)
        elif 'only_pep' in rla_types:
            final_tensor = F.pad(pep_tensor, (0, MAX_LEN - pep_tensor.shape[0]), mode='constant', value=0)
        elif 'pep_interweave' in rla_types:
            pep_weave = F.pad(torch.stack((pep_complex_tensor, pep_tensor)).view(pep_len*2), (0, max_chain_len - 2*pep_len), mode='constant', value=0)
            prot_final = F.pad(prot_complex_tensor, (0, max_chain_len - prot_len), mode='constant', value=0)
            final_tensor = torch.cat([pep_weave, prot_final])
        else:
            final_tensor = F.pad(complex_tensor, (0, MAX_LEN - (prot_len + pep_len)), mode='constant', value=0)
        data_tensor.append(final_tensor)
        data_coords.append(F.pad(coord_data[id_][0], (0, 0, 0, MAX_LEN - (prot_len + pep_len))))
        mask[final_tensor != 0] = 1
        data_mask.append(mask)

    data_tensor = torch.stack(data_tensor, dim=0)
    data_coords = torch.stack(data_coords, dim=0)
    data_mask = torch.stack(data_mask, dim=0)

    return data_tensor, data_coords, data_mask

def run_iter_gnn(base_data, pep_data, prot_data, df, binder_model, optimizer, loss_fn, task, rla_types, dev, grad, coord_data=None, batch_size=-1):
    if grad:
        torch.set_grad_enabled(True)
        binder_model.train()
    else:
        torch.set_grad_enabled(False)
        binder_model.eval()

    # try:
    #     for p in binder_model.h_V.parameters():
    #         p.requires_grad_(False)
    # except Exception as e:
    #     for p in binder_model.W_V.parameters():
    #         p.requires_grad_(False)

    # Subsample df and dataset to get balanced batch
    df_pos_ids = df[df['binder_4000_nm']]['description'].values
    df_all_neg_ids = df[~df['binder_4000_nm']]['description'].values
    max_chain_len = MAX_LEN // 2
    for _ in range(100):
        np.random.shuffle(df_all_neg_ids)
        df_neg_ids = df_all_neg_ids[:len(df_pos_ids)]

        
        pos_tensor, pos_coords, pos_mask = package_data_gnn(df_pos_ids, pep_data, prot_data, base_data, rla_types, max_chain_len)
        pos_labels = torch.ones(pos_tensor.shape[0])
        neg_tensor, neg_coords, neg_mask = package_data_gnn(df_neg_ids, pep_data, prot_data, base_data, rla_types, max_chain_len)
        neg_labels = torch.zeros(neg_tensor.shape[0])
        
        batch_tensor = torch.cat((pos_tensor, neg_tensor), dim=0)
        # batch_tensor = (batch_tensor - torch.mean(batch_tensor)) / torch.std(batch_tensor)
        batch_coords = torch.cat((pos_coords, neg_coords), dim=0)
        batch_mask = torch.cat((pos_mask, neg_mask), dim=0).to(dtype=torch.float64)
        indices = torch.randperm(batch_tensor.shape[0])
        batch_tensor = batch_tensor[indices]
        batch_mask = batch_mask[indices]
        batch_labels = torch.cat((pos_labels, neg_labels), dim=0)
        batch_labels = batch_labels[indices].to(device=dev, dtype=torch.float64)
        batch_coords = batch_coords[indices].to(device=dev, dtype=torch.float64)
        binder_model = binder_model.to(dtype=torch.float64)

        # Pass batch residue-level RLA scores through binder_model to get probabilities
        # batch_probs = []
        # for tens, coords, mask in zip(batch_tensor, batch_coords, batch_mask):
        #     batch_probs.append(binder_model(tens, coords, batch_mask).to(device=dev, dtype=torch.float64))
        # batch_probs = torch.cat(batch_probs)
        # Apply F.cross_entropy to get loss
        # Pass batch residue-level RLA scores through binder_model to get probabilities
        num_batches = math.ceil(batch_tensor.shape[0] / batch_size)
        for i in range(num_batches):
            cur_batch_tensor = batch_tensor[i*batch_size:(i+1)*batch_size]
            cur_batch_labels = batch_labels[i*batch_size:(i+1)*batch_size]
            cur_batch_mask = batch_mask[i*batch_size:(i+1)*batch_size]
            cur_batch_coords = batch_coords[i*batch_size:(i+1)*batch_size]
            batch_probs = binder_model(cur_batch_tensor.to(device=dev), cur_batch_coords.to(device=dev), cur_batch_mask.to(device=dev))

            # Apply F.cross_entropy to get loss
            # print(cur_batch_tensor)
            # print(batch_probs)
            # print(cur_batch_labels)
            # for n,p in binder_model.named_parameters():
            #     print(n, p.requires_grad)
            # raise ValueError
            loss = loss_fn(batch_probs, cur_batch_labels)
            if grad:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return loss.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train binder predictor!')
    parser.add_argument('--run_dir', help='Directory to save training results', required=True)
    parser.add_argument('--train_targets', help='Train target protein', required=True)
    parser.add_argument('--val_targets', help='Val target protein', required=True)
    parser.add_argument('--wds_path', help='Path to webdataset data', default='/data1/groups/keating_madry/wds')
    parser.add_argument('--results_dir', help='Path to RLA results and structure stats', default='/home/gridsan/fbirnbaum/joint-protein-embs/baker_results/', type=str)
    parser.add_argument('--epochs', help='Number of train epochs', default=100, type=int)
    parser.add_argument('--lr', help='Learning rate', default=1e-5, type=float)
    parser.add_argument('--regularization', help='Regularization', default=1e-3, type=float)
    parser.add_argument('--task', help='Type of task to learn', default='binary', type=str)
    parser.add_argument('--train_cut', help='Percent of data to use for training if same target training', default=0.8, type=float)
    parser.add_argument('--val_cut', help='Percent of data to use for validation if same target training', default=0.1, type=float)
    parser.add_argument('--early_stopping', help='Whether to use early stopping', default=False, type=bool)
    parser.add_argument('--split_target', help='Whether to trian/val split target data', default=False, type=bool)
    parser.add_argument('--split_scaff', help='Whether to split based on scaffold', default=True)
    parser.add_argument('--inter_res', help='Whether to only use interaction residues', default='all')
    parser.add_argument('--in_features', help='Model layer input sizes', default='1000,512,128,32', type=str)
    parser.add_argument('--out_features', help='Model layer output sizes', default='512,128,32,1', type=str)
    parser.add_argument('--num_layers', help='Number of linear layers', default=3, type=int)
    parser.add_argument('--n_heads', help='Number of attention heads', default=1, type=int)
    parser.add_argument('--dropout', help='Dropout level', default=0.0, type=float)
    parser.add_argument('--rla_types', help='Types of RLA scores to use', default='complex')
    parser.add_argument('--max_len', help='Padded max length', default=1000, type=int)
    parser.add_argument('--coordinator_hparams', help='Location of coord parameters', default='/home/gridsan/fbirnbaum/joint-protein-embs/terminator_configs/coordinator_broken_merge.json', type=str)
    parser.add_argument('--model_type', help='Type of model', default='linear', type=str)
    parser.add_argument('--batch_size', help='Batch size', default=-1, type=int)
    parser.add_argument('--use_dummy', help='Whether to use dummy node for GNN/Transformer', default=False, type=bool)
    parser.add_argument('--data_file', help='Raw binding data', default='/data1/groups/keating_madry/baker_designed_binders/all_data/retrospective_analysis/af2_rmsd_graphs1_data.sc')
    parser.add_argument('--all_name', help='Name for all RLA score file', default='all_res_scores_k_30_new_blacklist_wpep_weight_diff_all_noweight')
    parser.add_argument('--pep_name', help='Name for pep RLA score file', default='all_res_scores_k_30_new_blacklist_wpep_weight_diff_pep_all_noweight')
    parser.add_argument('--prot_name', help='Name for prot RLA score file', default='all_res_scores_k_30_new_blacklist_wpep_weight_diff_prot_all_noweight')
    
    parser.add_argument('--dev', help='Device to train on', default='cuda:0', type=str)
    args = parser.parse_args()
    if not os.path.exists(args.run_dir):
        os.mkdir(args.run_dir)
    print("Run dir: ", args.run_dir)
    print("Training on: ", args.train_targets)
    print("Val on : ", args.val_targets)
    print("Learning rate: ", args.lr)
    print("Regularization: ", args.regularization)
    print("Early stopping: ", args.early_stopping)
    print("Split target: ", args.split_target)
    print("Split scaff: ", args.split_scaff)
    print("Inter res: ", args.inter_res)
    print("Train cut: ", args.train_cut)
    print("Val cut: ", args.val_cut)
    print("In features: ", args.in_features)
    print("Out features: ", args.out_features)
    print("Num layers: ", args.num_layers)
    print("Num heads: ", args.n_heads)
    print("Dropout: ", args.dropout)
    print("RLA types: ", args.rla_types)
    print("Max length: ", args.max_len)
    print("COORD hparams: ", args.coordinator_hparams)
    print("Results dir: ", args.results_dir)
    print("Task: ", args.task)
    print("Model type: ", args.model_type)
    print("Batch size: ", args.batch_size)
    print("Use dummy: ", args.use_dummy)
    print("Data file: ", args.data_file)

    print("Dev: ", args.dev)

    MAX_LEN = args.max_len
    args.in_features = args.in_features.split(',')
    args.in_features = [int(feat) for feat in args.in_features]
    args.out_features = args.out_features.split(',')
    args.out_features = [int(feat) for feat in args.out_features]

    df = pd.read_csv(args.data_file, sep=' ')

    train_base_datas = []
    train_pep_datas = []
    train_prot_datas = []
    train_dfs = []
    args.train_targets = args.train_targets.split(',')
    val_base_datas = []
    val_pep_datas = []
    val_prot_datas = []
    val_dfs = []
    args.val_targets = args.val_targets.split(',')


    for target in args.train_targets:

        with open(f'{args.results_dir}/{target}_{args.all_name}.json', 'r') as f:
            train_base_datas.append(json.load(f))
        with open(f'{args.results_dir}/{target}_{args.pep_name}.json', 'r') as f:
            train_pep_datas.append(json.load(f))
        with open(f'{args.results_dir}/{target}_{args.prot_name}.json', 'r') as f:
            train_prot_datas.append(json.load(f))
        train_dfs.append(df[df['target'].str.contains(target)])
    if args.split_target and args.train_targets == args.val_targets:
        print('splitting by target')
        for i, target in enumerate(args.train_targets):
            pos_df = train_dfs[i][train_dfs[i]['binder_4000_nm']].sort_values(by='description')
            neg_df = train_dfs[i][~train_dfs[i]['binder_4000_nm']].sort_values(by='description')
            train_pos_cutoff = int(len(pos_df) * args.train_cut)
            train_neg_cutoff = int(len(neg_df) * args.train_cut)
            val_pos_cutoff = int(len(pos_df) * (args.train_cut + args.val_cut))
            val_neg_cutoff = int(len(neg_df) * (args.train_cut + args.val_cut))
            
            train_dfs[i] = pd.concat([pos_df.head(train_pos_cutoff), neg_df.head(train_neg_cutoff)])
            val_dfs.append(pd.concat([pos_df.iloc[train_pos_cutoff:val_pos_cutoff], neg_df.iloc[train_neg_cutoff:val_neg_cutoff]]))
        val_base_datas = train_base_datas
        val_pep_datas = train_pep_datas
        val_prot_datas = train_prot_datas
    elif args.split_scaff and args.train_targets == args.val_targets:
        for i, target in enumerate(args.train_targets):
            val_dfs.append(train_dfs[i][train_dfs[i]['scaff_class'] == HOLD_OUTS[target]])
            train_dfs[i] = train_dfs[i][train_dfs[i]['scaff_class'] != HOLD_OUTS[target]]
        val_base_datas = train_base_datas
        val_pep_datas = train_pep_datas
        val_prot_datas = train_prot_datas
    else:
        for target in args.val_targets:
            with open(f'{args.results_dir}/{target}_{args.all_name}.json', 'r') as f:
                val_base_datas.append(json.load(f))
            with open(f'{args.results_dir}/{target}_{args.pep_name}.json', 'r') as f:
                val_pep_datas.append(json.load(f))
            with open(f'{args.results_dir}/{target}_{args.prot_name}.json', 'r') as f:
                val_prot_datas.append(json.load(f))
            val_dfs.append(df[df['target'].str.contains(target)])

    progress = tqdm(total=args.epochs)
    if args.model_type == 'linear':
        binder_model = BinderModel(in_features=args.in_features, out_features=args.out_features, num_layers=args.num_layers, dropout=args.dropout, predict_prob=(args.task == 'binary')).to(device=args.dev)
    elif args.model_type == 'gnn':
        print('usign gnn x)')
        binder_model = BinderGNN(node_in_dim=1, edge_in_dim=1, node_hidden_dim=args.in_features[0], edge_hidden_dim=args.in_features[0], num_hidden_layers=args.num_layers, dropout=0.0, predict_prob=(args.task == 'binary'), use_dummy=args.use_dummy).to(device=args.dev)
    elif args.model_type == 'transformer':
        print('using transformer :)')
        binder_model = BinderTransformerModel(d_in=1, d_model=args.in_features[0], nhead=args.n_heads, num_encoder_layers=args.num_layers, dim_feedforward=args.in_features[0], dropout=args.dropout, predict_prob=True, batch_first=True, use_dummy=args.use_dummy).to(device=args.dev)

    optimizer = torch.optim.Adam(binder_model.parameters(), lr=args.lr, weight_decay=args.regularization)
    
    writer = SummaryWriter(log_dir=os.path.join(args.run_dir, 'tensorboard'))
    training_curves = {"train_loss": [], "val_loss": []}
    best_val_loss = None
    epochs_since_improvement = 0
    if args.task == 'binary':
        loss_fn = nn.BCELoss()
    elif args.task == 'regression':
        loss_fn = nn.MSELoss()
    coordinator_params = data_utils.get_coordinator_params(args.coordinator_hparams)

    target_coord_data = {}
    for i, target in enumerate(args.train_targets):
        if args.inter_res == 'all':
            coord_data_path = f'{args.results_dir}/{target}_struct_stats_all.pickle'
        elif args.inter_res == 'inter':
            coord_data_path = f'{args.results_dir}/{target}_struct_stats.pickle'
        elif args.inter_res == 'coord':
            coord_data_path = f'{args.results_dir}/{target}_struct_coords.pickle'
        else:
            coord_data_path = ''
        if len(coord_data_path) > 0:
            with open(coord_data_path, 'rb') as f:
                coord_data = pickle.load(f)
            target_coord_data[target] = coord_data
        else:
            target_coord_data[target] = None

    if args.inter_res == 'none':
        inter_fun = run_iter_old
    elif args.inter_res in ['all', 'inter']:
        inter_fun = run_iter
    else:
        inter_fun = run_iter_gnn


    for epoch in range(args.epochs):

        # Train iter
        train_losses = {}
        for i, target in enumerate(args.train_targets):
            train_loss = inter_fun(train_base_datas[i], train_pep_datas[i], train_prot_datas[i], train_dfs[i], binder_model, optimizer, loss_fn, args.task, args.rla_types, args.dev, grad=True, coord_data=target_coord_data[target], batch_size=args.batch_size)
            train_losses[target] = train_loss
        # Val iter
        val_losses = {}
        for i, target in enumerate(args.val_targets):
            val_loss = inter_fun(val_base_datas[i], val_pep_datas[i], val_prot_datas[i], val_dfs[i], binder_model, optimizer, loss_fn, args.task, args.rla_types, args.dev, grad=False, coord_data=target_coord_data[target], batch_size=args.batch_size)
            val_losses[target] = val_loss

        # Upkeep        
        progress.update(1)
        progress.refresh()
        progress.set_description_str(f'train loss {train_losses} | val loss {val_losses}')
        writer.add_scalar('train loss', np.mean(list(train_losses.values())), epoch)
        writer.add_scalar('val loss', np.mean(list(val_losses.values())), epoch)
        training_curves["train_loss"].append(np.mean(list(train_losses.values())))
        training_curves["val_loss"].append(np.mean(list(val_losses.values())))
        # Save a state checkpoint
        checkpoint_state = {
            'epoch': epoch,
            'state_dict': binder_model.state_dict(),
            'val_loss': np.mean(list(val_losses.values())),
            'optimizer_state': optimizer.state_dict(),
            'training_curves': training_curves
        }
        torch.save(checkpoint_state, os.path.join(args.run_dir, 'net_last_checkpoint.pt'))

        # Early stopping
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            torch.save(checkpoint_state, os.path.join(args.run_dir, 'net_best_checkpoint.pt'))
        else:
            epochs_since_improvement += 1

        if args.early_stopping and epochs_since_improvement >= 15:
            print('Early stopping')
            break