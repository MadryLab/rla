""" Util functions useful in TERMinator modules """
import sys
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# pylint: disable=no-member

# batchify functions



def pad_sequence_12(sequences, padding_value=0):
    """Given a sequence of tensors, batch them together by pads both dims 1 and 2 to max length.

    Args
    ----
    sequences : list of torch.Tensor
        Sequence of tensors with number of axes `N >= 2`
    padding value : int, default=0
        What value to pad the tensors with

    Returns
    -------
    out_tensor : torch.Tensor
        Batched tensor with shape (n_batch, max_dim1, max_dim2, ...)
    """
    n_batches = len(sequences)
    out_dims = list(sequences[0].size())
    dim1, dim2 = 0, 1
    max_dim1 = max([s.size(dim1) for s in sequences])
    max_dim2 = max([s.size(dim2) for s in sequences])
    out_dims[dim1] = max_dim1
    out_dims[dim2] = max_dim2
    out_dims = [n_batches] + out_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        len1 = tensor.size(0)
        len2 = tensor.size(1)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :len1, :len2, ...] = tensor

    return out_tensor


def batchify(batched_flat_terms, term_lens):
    """ Take a flat representation of TERM information and batch them into a stacked representation.

    In the TERM information condensor, TERM information is initially stored by concatenating all
    TERM tensors side by side in one dimension. However, for message passing, it's convenient to batch
    these TERMs by splitting them and stacking them in a new dimension.

    Args
    ----
    batched_flat_terms : torch.Tensor
        Tensor with shape :code:`(n_batch, sum_term_len, ...)`
    term_lens : list of (list of int)
        Length of each TERM per protein

    Returns
    -------
    batchify_terms : torch.Tensor
        Tensor with shape :code:`(n_batch, max_num_terms, max_term_len, ...)`
    """
    n_batches = batched_flat_terms.shape[0]
    flat_terms = torch.unbind(batched_flat_terms)
    list_terms = [torch.split(flat_terms[i], term_lens[i]) for i in range(n_batches)]
    padded_terms = [pad_sequence(terms) for terms in list_terms]
    padded_terms = [term.transpose(0, 1) for term in padded_terms]
    batchify_terms = pad_sequence_12(padded_terms)
    return batchify_terms

# Extract kNN info
def extract_knn(X, mask, eps, top_k):
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)
        return mask_2D, D_neighbors, E_idx

# # Extract chain break and end info
# def parse_chain_ends(X_batch, mask_batch, chain_idx, type='replace'):
#     all_begin = []
#     all_end = []
#     all_full_mask = []
#     for i, X in enumerate(X_batch):
#         _, seq_len = torch.unique_consecutive(mask_batch[i], return_counts=True)
#         _, chain_lens = torch.unique_consecutive(chain_idx[i,:seq_len[0]], return_counts=True)
#         assert len(chain_lens) == max(chain_idx[i]) + 1

#         if type == 'mask':
#             begin = torch.ones(X.shape[0])
#             end = torch.ones(X.shape[0])
#         elif type == 'replace':
#             begin = torch.arange(X.shape[0])
#             end = torch.arange(X.shape[0])
#         prev_cl = 0
#         full_mask = torch.ones(X.shape[0])
#         for cl in chain_lens:
#             if type == 'mask':
#                 begin[prev_cl] = 0
#                 end[prev_cl+cl-1] = 0
#             elif type == 'replace':
#                 begin[prev_cl] = min(X.shape[0]-1, prev_cl+1)
#                 end[prev_cl+cl-1] = max(0, prev_cl+cl-2)
#             if cl == 1:
#                 full_mask[prev_cl] = 0
#             prev_cl += cl
#         all_begin.append(begin)
#         all_end.append(end)
#         all_full_mask.append(full_mask)

#     begin = pad_sequence(all_begin, batch_first=True).to(X_batch.device)
#     end = pad_sequence(all_end, batch_first=True).to(X_batch.device)
#     full_mask = pad_sequence(all_full_mask, batch_first=True).to(X_batch.device)
#     return begin, end, full_mask


# gather and cat functions
# struct level

def extract_idxs(E_idx, mask):
    ref_indexes = torch.arange(E_idx.shape[0])
    ref_indexes = ref_indexes.unsqueeze(-1)
    ref_indexes = ref_indexes.expand(E_idx.shape)
    ref_indexes = ref_indexes.to(E_idx.device)
    max_length = int(10**(np.ceil(np.log10(E_idx.shape[0]))))
    opt1 = torch.from_numpy(E_idx.numpy() / max_length + ref_indexes.numpy())
    opt2 = torch.from_numpy(E_idx.numpy() + ref_indexes.numpy() / max_length)
    E_idx_paired = torch.min(opt1, opt2)
    del opt1, opt2
    # E_idx_paired = torch.min(E_idx / max_length + ref_indexes, E_idx + ref_indexes / max_length)
    E_idx_paired_flat = torch.flatten(E_idx_paired)
    out, count = torch.unique(E_idx_paired_flat, sorted=True, return_inverse=False, return_counts=True, dim=0)

    out_dup = out[count > 1]
    o_size = out_dup.numel()
    v_size = E_idx_paired_flat.numel()

    o_expand = out_dup.unsqueeze(1).expand(o_size, v_size) 
    v_expand = E_idx_paired_flat.unsqueeze(0).expand(o_size, v_size)
    result = (o_expand - v_expand == 0).nonzero()[:,1]
    idxs_to_dup = result[::2]
    idxs_to_remove = result[1::2]

    result_singles = torch.arange(len(E_idx_paired_flat)).type(idxs_to_dup.dtype)
    base = -1 * torch.ones(len(E_idx_paired_flat)).type(idxs_to_dup.dtype)
    result_singles = result_singles.scatter_(0, idxs_to_dup, base)
    result_singles = result_singles.scatter_(0, idxs_to_remove, base)
    idxs_singles = result_singles[result_singles > 0]
        
    inds_reduce, mask_reduced_combs = per_node_to_all_comb_inds(idxs_to_remove, E_idx_paired.shape)
    inds_expand = all_comb_to_per_node_inds(idxs_to_remove, idxs_to_dup, E_idx_paired.shape, inds_reduce.shape)
    mask = mask[0].unsqueeze(1)
    mask_reduced_nodes = mask.expand(E_idx_paired.shape)
    mask_reduced_nodes = per_node_to_all_comb(mask_reduced_nodes, inds_reduce, batched=False)
    mask_reduced = torch.multiply(mask_reduced_combs, mask_reduced_nodes)

    return inds_reduce, inds_expand, idxs_to_remove, idxs_to_dup, idxs_singles, mask_reduced.to(torch.bool) 

def sync_inds_shape(inds, shape):
    i = len(inds.shape)
    while len(shape) > len(inds.shape):
        inds = torch.unsqueeze(inds, -1)
        new_shape = list(inds.shape)
        new_shape[i] = shape[i]
        inds = inds.expand(new_shape)
        i += 1
    return inds

def per_node_to_all_comb_inds(idxs_to_remove, per_node_shape):
    base = -1*torch.ones(per_node_shape[0] * per_node_shape[1], dtype=torch.int64)
    inds = torch.arange(len(base), dtype=torch.int64).scatter_(0, idxs_to_remove, base)
    all_inds = inds[inds > -1]
    mask = torch.logical_not(torch.isnan(all_inds))
    all_inds = torch.nan_to_num(all_inds)
    all_inds = all_inds.type(torch.int64)
    return all_inds, mask

def per_node_to_all_comb(per_node_tensor, inds_reduce, batched=True):
    if batched:
        begin_dim = 1
    else:
        begin_dim = 0

    all_comb_tensor = torch.flatten(per_node_tensor, begin_dim, begin_dim+1)
    inds_reduce = sync_inds_shape(inds_reduce, all_comb_tensor.shape)
    all_comb_tensor = torch.gather(all_comb_tensor, begin_dim, inds_reduce)
    return all_comb_tensor
    
def all_comb_to_per_node_inds(idxs_to_remove, idxs_to_dup, per_node_shape, all_combs_shape):
    idxs_to_remove = idxs_to_remove.type(torch.int64)
    
    base = -1*torch.ones(per_node_shape[0] * per_node_shape[1], dtype=torch.int64)
    inds = torch.arange(len(base), dtype=torch.int64)
    idx_to_keep = inds.scatter_(0, idxs_to_remove, base)
    idx_to_keep = idx_to_keep[idx_to_keep > -1]

    small_inds = torch.arange(all_combs_shape[0], dtype=base.dtype)
    inds = base.scatter_(0, idx_to_keep, small_inds)
    dup_inds = torch.gather(inds, 0, idxs_to_dup)
    all_inds = inds.scatter_(0, idxs_to_remove, dup_inds).type(torch.int64)
    return all_inds

def all_comb_to_per_node(all_comb_tensor, inds_expand, per_node_shape, batched=True):
    if batched:
        begin_dim = 1
    else:
        begin_dim = 0
    inds_expand = sync_inds_shape(inds_expand, all_comb_tensor.shape)
    per_node_tensor = torch.gather(all_comb_tensor, begin_dim, inds_expand)
    per_node_tensor = per_node_tensor.reshape(per_node_shape)
    return per_node_tensor

def all_comb_to_per_node_transpose(all_comb_tensor, inds_expand, inds_transpose, per_node_shape, batched=True):
    if batched:
        begin_dim = 1
    else:
        begin_dim = 0
    inds_expand = sync_inds_shape(inds_expand, all_comb_tensor.shape)
    per_node_tensor = torch.gather(all_comb_tensor, begin_dim, inds_expand)
    inds_transpose = sync_inds_shape(inds_transpose, per_node_tensor.shape)
    per_node_tensor_transpose = torch.clone(per_node_tensor.transpose(-2, -1))
    per_node_tensor_transpose = torch.gather(per_node_tensor_transpose, begin_dim, inds_transpose)
    per_node_tensor = per_node_tensor.scatter_(begin_dim, inds_transpose, per_node_tensor_transpose)
    per_node_tensor = per_node_tensor.reshape(per_node_shape)
    return per_node_tensor

def average_duplicates(data, inds1, inds2):
    orig_shape = data.shape
    data = torch.flatten(data, 1, 2)
    inds1 = sync_inds_shape(inds1, data.shape)
    inds2 = sync_inds_shape(inds2, data.shape)
    data1 = torch.gather(data, 1, inds1)
    data2 = torch.gather(data, 1, inds2)
    base = torch.zeros(data.shape).to(data.device)
    base = base.scatter_(1, inds1, data2)
    base = base.scatter_(1, inds2, data1)
    mask = base != 0
    count = mask + 1
    data = torch.div(data + base, count)
    return data.view(orig_shape)

def concatenate_duplicates(data, inds1, inds2, inds3):
    orig_shape = data.shape
    data = torch.flatten(data, 1, 2)
    inds1_gather = sync_inds_shape(inds1, data.shape)
    inds2_gather = sync_inds_shape(inds2, data.shape)
    inds3_gather = sync_inds_shape(inds3, data.shape)
    data1 = torch.gather(data, 1, inds1_gather)
    data2 = torch.gather(data, 1, inds2_gather)
    data3 = torch.gather(data, 1, inds3_gather)
    data_12_cat = torch.cat((data1, data2), -1)
    data_3_cat = torch.cat((data3, data3), -1)
    new_data = torch.zeros(data.shape[:-1] + (2*orig_shape[-1],)).to(data.device)
    inds1_scatter = sync_inds_shape(inds1, new_data.shape)
    inds2_scatter = sync_inds_shape(inds2, new_data.shape)
    inds3_scatter = sync_inds_shape(inds3, new_data.shape)
    new_data.scatter_(1, inds1_scatter, data_12_cat)
    new_data.scatter_(1, inds2_scatter, data_12_cat)
    new_data.scatter_(1, inds3_scatter, data_3_cat)
    return new_data.view(orig_shape[:-1] + (2*orig_shape[-1],))

def gather_edges(edges, neighbor_idx):
    """ Gather the edge features of the nearest neighbors.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    edges : torch.Tensor
        The edge features in dense form
        Shape: n_batch x n_res x n_res x n_hidden
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    edge_features : torch.Tensor
        The gathered edge features
        Shape : n_batch x n_res x k x n_hidden
    """
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    """ Gather node features of nearest neighbors.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_res x n_hidden
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    neighbor_features : torch.Tensor
        The gathered neighbor node features
        Shape : n_batch x n_res x k x n_hidden
    """
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """ Concatenate node features onto the ends of gathered edge features given kNN sparse edge indices

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    h_nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_res x n_hidden
    h_neighbors : torch.Tensor
        The gathered edge features
        Shape: n_batch x n_res x k x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    h_nn : torch.Tensor
        The gathered concatenated node and edge features
        Shape : n_batch x n_res x k x n_hidden
    """
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def cat_edge_endpoints(h_edges, h_nodes, E_idx, inds_expand = None, inds_transpose = None):
    """ Concatenate both node features onto the ends of gathered edge features given kNN sparse edge indices

    Args
    ----
    h_edges : torch.Tensor
        The gathered edge features
        Shape: n_batch x n_res x k x n_hidden
    h_nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_res x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    h_nn : torch.Tensor
        The gathered concatenated node and edge features
        Shape : n_batch x n_res x k x 3*n_hidden
    """
    # Reshape edge features if needed
    if inds_expand is not None and inds_transpose is not None:
        h_edges = all_comb_to_per_node_transpose(h_edges, inds_expand, inds_transpose, per_node_shape=h_nodes.shape[:2] + (E_idx.shape[-1],) + (h_edges.shape[-1],))

    # Neighbor indices E_idx [B,N,K]
    # Edge features h_edges [B,N,N,C]
    # Node features h_nodes [B,N,C]
    k = E_idx.shape[-1]

    h_i_idx = E_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, k).contiguous()
    h_j_idx = E_idx

    h_i = gather_nodes(h_nodes, h_i_idx)
    h_j = gather_nodes(h_nodes, h_j_idx)

    # output features [B, N, K, 3C]
    h_nn = torch.cat([h_i, h_j, h_edges], -1)
    return h_nn


def gather_pairEs(pairEs, neighbor_idx):
    """ Gather the pair energies features of the nearest neighbors.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    pairEs : torch.Tensor
        The pair energies in dense form
        Shape: n_batch x n_res x n_res x n_aa x n_aa
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    pairE_features : torch.Tensor
        The gathered pair energies
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    n_aa = pairEs.size(-1)
    neighbors = neighbor_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, n_aa, n_aa)
    pairE_features = torch.gather(pairEs, 2, neighbors)
    return pairE_features


# term level


def gather_term_nodes(nodes, neighbor_idx):
    """ Gather TERM node features of nearest neighbors.

    Adatped from https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_terms x n_res x n_hidden
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    neighbor_features : torch.Tensor
        The gathered neighbor node features
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    # Features [B,T,N,C] at Neighbor indices [B,T,N,K] => [B,T,N,K,C]
    # Flatten and expand indices per batch [B,T,N,K] => [B,T,NK] => [B,T,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], neighbor_idx.shape[1], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, -1, nodes.size(3))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 2, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:4] + [-1])
    return neighbor_features


def gather_term_edges(edges, neighbor_idx):
    """ Gather the TERM edge features of the nearest neighbors.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    edges : torch.Tensor
        The edge features in dense form
        Shape: n_batch x n_terms x n_res x n_res x n_hidden
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    edge_features : torch.Tensor
        The gathered edge features
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    # Features [B,T,N,N,C] at Neighbor indices [B,T,N,K] => Neighbor features [B,T,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 3, neighbors)
    return edge_features


def cat_term_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """ Concatenate node features onto the ends of gathered edge features given kNN sparse edge indices

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    h_nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_terms x n_res x n_hidden
    h_neighbors : torch.Tensor
        The gathered edge features
        Shape: n_batch x n_terms x n_res x k x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    h_nn : torch.Tensor
        The gathered concatenated node and edge features
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    h_nodes = gather_term_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def cat_term_edge_endpoints(h_edges, h_nodes, E_idx):
    """ Concatenate both node features onto the ends of gathered edge features given kNN sparse edge indices

    Args
    ----
    h_edges : torch.Tensor
        The gathered edge features
        Shape: n_batch x n_terms x n_res x k x n_hidden
    h_nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_terms x n_res x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    h_nn : torch.Tensor
        The gathered concatenated node and edge features
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    # Neighbor indices E_idx [B,T,N,K]
    # Edge features h_edges [B,T,N,N,C]
    # Node features h_nodes [B,T,N,C]
    k = E_idx.shape[-1]

    h_i_idx = E_idx[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, k).contiguous()
    h_j_idx = E_idx

    h_i = gather_term_nodes(h_nodes, h_i_idx)
    h_j = gather_term_nodes(h_nodes, h_j_idx)

    # e_ij = gather_edges(h_edges, E_idx)
    e_ij = h_edges

    # output features [B, T, N, K, 3C]
    h_nn = torch.cat([h_i, h_j, e_ij], -1)
    return h_nn



# merge edge fns


def merge_duplicate_edges(h_E_update, E_idx):
    """ Average embeddings across bidirectional edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings.

    Args
    ----
    h_E_update : torch.Tensor
        Update tensor for edges embeddings in kNN sparse form
        Shape : n_batch x n_res x k x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    merged_E_updates : torch.Tensor
        Edge update with merged updates for bidirectional edges
        Shape : n_batch x n_res x k x n_hidden
    """

    seq_lens = torch.ones(h_E_update.shape[0]).long().to(h_E_update.device) * h_E_update.shape[1]
    h_dim = h_E_update.shape[-1]
    h_E_geometric = h_E_update.view([-1, h_dim])
    split_E_idxs = torch.unbind(E_idx)
    offset = [seq_lens[:i].sum() for i in range(len(seq_lens))]
    split_E_idxs = [e.to(h_E_update.device) + o for e, o in zip(split_E_idxs, offset)]
    edge_index_row = torch.cat([e.view(-1) for e in split_E_idxs], dim=0)
    edge_index_col = torch.repeat_interleave(torch.arange(edge_index_row.shape[0] // 30), 30).to(h_E_update.device)
    edge_index = torch.stack([edge_index_row, edge_index_col])
    merge = merge_duplicate_edges_geometric(h_E_geometric, edge_index)
    merge = merge.view(h_E_update.shape)

    # dev = h_E_update.device
    # n_batch, n_nodes, _, hidden_dim = h_E_update.shape
    # # collect edges into NxN tensor shape
    # collection = torch.zeros((n_batch, n_nodes, n_nodes, hidden_dim)).to(dev)
    # neighbor_idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, hidden_dim).to(dev)
    # collection.scatter_(2, neighbor_idx, h_E_update)
    # # transpose to get same edge in reverse direction
    # collection = collection.transpose(1, 2)
    # # gather reverse edges
    # reverse_E_update = gather_edges(collection, E_idx)
    # # average h_E_update and reverse_E_update at non-zero positions
    # merged_E_updates = torch.where(reverse_E_update != 0, (h_E_update + reverse_E_update) / 2, h_E_update)
    # assert (merge == merged_E_updates).all()

    return merge


def merge_duplicate_edges_geometric(h_E_update, edge_index):
    """ Average embeddings across bidirectional edges for Torch Geometric graphs

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings.

    Args
    ----
    h_E_update : torch.Tensor
        Update tensor for edges embeddings in Torch Geometric sparse form
        Shape : n_edge x n_hidden
    edge_index : torch.LongTensor
        Torch Geometric sparse edge indices
        Shape : 2 x n_edge

    Returns
    -------
    merged_E_updates : torch.Tensor
        Edge update with merged updates for bidirectional edges
        Shape : n_edge x n_hidden
    """
    num_nodes = edge_index.max() + 1
    row_idx = edge_index[0] + edge_index[1] * num_nodes
    col_idx = edge_index[1] + edge_index[0] * num_nodes
    internal_idx = torch.arange(edge_index.shape[1])

    mapping = torch.zeros(max(row_idx.max(), col_idx.max()) + 1).long() - 1
    mapping[col_idx] = internal_idx

    reverse_idx = mapping[row_idx]
    mask = (reverse_idx >= 0)
    reverse_idx = reverse_idx[mask]

    reverse_h_E = h_E_update[mask].clone()
    h_E_update[reverse_idx] = (h_E_update[reverse_idx].clone() + reverse_h_E)/2

    return h_E_update


def merge_duplicate_term_edges(h_E_update, E_idx):
    """ Average embeddings across bidirectional TERM edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings.

    Args
    ----
    h_E_update : torch.Tensor
        Update tensor for edges embeddings in kNN sparse form
        Shape : n_batch x n_terms x n_res x k x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    merged_E_updates : torch.Tensor
        Edge update with merged updates for bidirectional edges
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    dev = h_E_update.device
    n_batch, n_terms, n_aa, _, hidden_dim = h_E_update.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, n_terms, n_aa, n_aa, hidden_dim)).to(dev)
    neighbor_idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, -1, hidden_dim).to(dev)
    collection.scatter_(3, neighbor_idx, h_E_update)
    # transpose to get same edge in reverse direction
    collection = collection.transpose(2, 3)
    # gather reverse edges
    reverse_E_update = gather_term_edges(collection, E_idx)
    # average h_E_update and reverse_E_update at non-zero positions
    merged_E_updates = torch.where(reverse_E_update != 0, (h_E_update + reverse_E_update) / 2, h_E_update)
    return merged_E_updates


def merge_duplicate_pairE(h_E, E_idx):
    """ Average pair energy tables across bidirectional edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    try:
        seq_lens = torch.ones(h_E.shape[0]).long().to(h_E.device) * h_E.shape[1]
        h_E_geometric = h_E.view([-1, 400])
        split_E_idxs = torch.unbind(E_idx)
        offset = [seq_lens[:i].sum() for i in range(len(seq_lens))]
        split_E_idxs = [e.to(h_E.device) + o for e, o in zip(split_E_idxs, offset)]
        edge_index_row = torch.cat([e.view(-1) for e in split_E_idxs], dim=0)
        edge_index_col = torch.repeat_interleave(torch.arange(edge_index_row.shape[0] // 30), 30).to(h_E.device)
        edge_index = torch.stack([edge_index_row, edge_index_col])
        merge = merge_duplicate_pairE_geometric(h_E_geometric, edge_index)
        merge = merge.view(h_E.shape)
        #old_merge = merge_duplicate_pairE_dense(h_E, E_idx)
        #assert (old_merge == merge).all(), (old_merge, merge)

        return merge
    except RuntimeError as err:
        print(err, file=sys.stderr)
        print("We're handling this error as if it's an out-of-memory error", file=sys.stderr)
        torch.cuda.empty_cache()  # this is probably unnecessary but just in case
        return merge_duplicate_pairE_sparse(h_E, E_idx)


def merge_duplicate_pairE_dense(h_E, E_idx):
    """ Dense method to average pair energy tables across bidirectional edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    dev = h_E.device
    n_batch, n_nodes, _, n_aa, _ = h_E.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, n_nodes, n_nodes, n_aa, n_aa)).to(dev)
    neighbor_idx = E_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, n_aa, n_aa).to(dev)
    collection.scatter_(2, neighbor_idx, h_E)
    # transpose to get same edge in reverse direction
    collection = collection.transpose(1, 2)
    # transpose each pair energy table as well
    collection = collection.transpose(-2, -1)
    # gather reverse edges
    reverse_E = gather_pairEs(collection, E_idx)
    # average h_E and reverse_E at non-zero positions
    merged_E = torch.where(reverse_E != 0, (h_E + reverse_E) / 2, h_E)
    return merged_E


# TODO: rigorous test that this is equiv to the dense version
def merge_duplicate_pairE_sparse(h_E, E_idx):
    """ Sparse method to average pair energy tables across bidirectional edges.

    Note: This method involves a significant slowdown so it's only worth using if memory is an issue.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    dev = h_E.device
    n_batch, n_nodes, k, n_aa, _ = h_E.shape
    # convert etab into a sparse etab
    # self idx of the edge
    ref_idx = E_idx[:, :, 0:1].expand(-1, -1, k)
    # sparse idx
    g_idx = torch.cat([E_idx.unsqueeze(1), ref_idx.unsqueeze(1)], dim=1)
    sparse_idx = g_idx.view([n_batch, 2, -1])
    # generate a 1D idx for the forward and backward direction
    scaler = torch.ones_like(sparse_idx).to(dev)
    scaler = scaler * n_nodes
    scaler_f = scaler
    scaler_f[:, 0] = 1
    scaler_r = torch.flip(scaler_f, [1])
    batch_offset = torch.arange(n_batch).unsqueeze(-1).expand([-1, n_nodes * k]) * n_nodes * k
    batch_offset = batch_offset.to(dev)
    sparse_idx_f = torch.sum(scaler_f * sparse_idx, 1) + batch_offset
    flat_idx_f = sparse_idx_f.view([-1])
    sparse_idx_r = torch.sum(scaler_r * sparse_idx, 1) + batch_offset
    flat_idx_r = sparse_idx_r.view([-1])
    # generate sparse tensors
    flat_h_E_f = h_E.view([n_batch * n_nodes * k, n_aa**2])
    reverse_h_E = h_E.transpose(-2, -1).contiguous()
    flat_h_E_r = reverse_h_E.view([n_batch * n_nodes * k, n_aa**2])
    sparse_etab_f = torch.sparse_coo_tensor(flat_idx_f.unsqueeze(0), flat_h_E_f,
                                            (n_batch * n_nodes * n_nodes, n_aa**2))
    count_f = torch.sparse_coo_tensor(flat_idx_f.unsqueeze(0), torch.ones_like(flat_idx_f),
                                      (n_batch * n_nodes * n_nodes, ))
    sparse_etab_r = torch.sparse_coo_tensor(flat_idx_r.unsqueeze(0), flat_h_E_r,
                                            (n_batch * n_nodes * n_nodes, n_aa**2))
    count_r = torch.sparse_coo_tensor(flat_idx_r.unsqueeze(0), torch.ones_like(flat_idx_r),
                                      (n_batch * n_nodes * n_nodes, ))
    # merge
    sparse_etab = sparse_etab_f + sparse_etab_r
    sparse_etab = sparse_etab.coalesce()
    count = count_f + count_r
    count = count.coalesce()

    # this step is very slow, but implementing something faster is probably a lot of work
    # requires pytorch 1.10 to be fast enough to be usable
    collect = sparse_etab.index_select(0, flat_idx_f).to_dense()
    weight = count.index_select(0, flat_idx_f).to_dense()

    flat_merged_etab = collect / weight.unsqueeze(-1)
    merged_etab = flat_merged_etab.view(h_E.shape)
    return merged_etab


def merge_duplicate_pairE_geometric(h_E, edge_index):
    """ Sparse method to average pair energy tables across bidirectional edges with Torch Geometric.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    This function assumes edge_index is sorted by columns, and will fail if
    this is not the case.

    Args
    ----
    h_E : torch.Tensor
        Pair energies in Torch Geometric sparse form
        Shape : n_edge x 400
    E_idx : torch.LongTensor
        Torch Geometric sparse edge indices
        Shape : 2 x n_edge

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_edge x 400
    """
    num_nodes = edge_index.max() + 1
    row_idx = edge_index[0] + edge_index[1] * num_nodes
    col_idx = edge_index[1] + edge_index[0] * num_nodes
    internal_idx = torch.arange(edge_index.shape[1])

    mapping = torch.zeros(max(row_idx.max(), col_idx.max()) + 1).long() - 1
    mapping[col_idx] = internal_idx

    reverse_idx = mapping[row_idx]
    mask = (reverse_idx >= 0)
    reverse_idx = reverse_idx[mask]

    reverse_h_E = h_E[mask]
    transpose_h_E = reverse_h_E.view([-1, 20, 20]).transpose(-1, -2).reshape([-1, 400])
    h_E[reverse_idx] = (h_E[reverse_idx] + transpose_h_E)/2

    return h_E


# edge aggregation fns


def aggregate_edges(edge_embeddings, E_idx, max_seq_len):
    """ Aggregate TERM edge embeddings into a sequence-level dense edge features tensor

    Args
    ----
    edge_embeddings : torch.Tensor
        TERM edge features tensor
        Shape : n_batch x n_terms x n_aa x n_neighbors x n_hidden
    E_idx : torch.LongTensor
        TERM edge indices
        Shape : n_batch x n_terms x n_aa x n_neighbors
    max_seq_len : int
        Max length of a sequence in the batch

    Returns
    -------
    torch.Tensor
        Dense sequence-level edge features
        Shape : n_batch x max_seq_len x max_seq_len x n_hidden
    """
    dev = edge_embeddings.device
    n_batch, _, _, n_neighbors, hidden_dim = edge_embeddings.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, max_seq_len, max_seq_len, hidden_dim)).to(dev)
    # edge the edge indecies
    self_idx = E_idx[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, n_neighbors)
    neighbor_idx = E_idx
    # tensor needed for accumulation
    layer = torch.arange(n_batch).view([n_batch, 1, 1, 1]).expand(neighbor_idx.shape).to(dev)
    # thicc index_put_
    collection.index_put_((layer, self_idx, neighbor_idx), edge_embeddings, accumulate=True)

    # we also need counts for averaging
    count = torch.zeros((n_batch, max_seq_len, max_seq_len)).to(dev)
    count_idx = torch.ones_like(neighbor_idx).float().to(dev)
    count.index_put_((layer, self_idx, neighbor_idx), count_idx, accumulate=True)

    # we need to set all 0s to 1s so we dont get nans
    count[count == 0] = 1

    return collection / count.unsqueeze(-1)
