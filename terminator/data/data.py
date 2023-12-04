"""Datasets and dataloaders for loading Coords.

This file contains dataset and dataloader classes
to be used when interacting with Coords.
"""
import glob
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
import torch_geometric
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
DIR = os.path.dirname(os.path.abspath(__file__))
assert DIR[0] == "/", "DIR should be an abspath"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(DIR)), 'terminator_utils', 'scripts', 'data', 'preprocessing'))
sys.path.insert(0, '../../terminator_utils/scripts/data/preprocessing')
sys.path.insert(0,os.path.join('/'.join(DIR.split('/')[0:-2]),'terminator_utils','scripts','data','preprocessing'))
from parseCoords import parseCoords
from terminator.utils.common import seq_to_ints
from terminator.models.layers.utils import extract_knn, extract_idxs, per_node_to_all_comb

# pylint: disable=no-member, not-callable


# Jing featurization functions


def _normalize(tensor, dim=-1):
    '''Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.'''
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.

    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].

    From https://github.com/jingraham/neurips19-graph-protein-design
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    rbf = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return rbf


def _dihedrals(X, eps=1e-7):
    """ Compute dihedral angles between residues given atomic backbone coordinates

    Args
    ----
    X : torch.FloatTensor
        Tensor specifying atomic backbone coordinates
        Shape: num_res x 4 x 3

    Returns
    -------
    D_features : torch.FloatTensor
        Dihedral angles, lifted to the 3-torus
        Shape: num_res x 7
    """
    # From https://github.com/jingraham/neurips19-graph-protein-design

    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def _positional_embeddings(edge_index, num_embeddings=16, dev='cpu'):
    """ Sinusoidally encode sequence distances for edges.

    Args
    ----
    edge_index : torch.LongTensor
        Edge indices for sparse representation of protein graph
        Shape: 2 x num_edges
    num_embeddings : int or None, default=128
        Dimensionality of sinusoidal embedding.

    Returns
    -------
    E : torch.FloatTensor
        Sinusoidal encoding of sequence distances
        Shape: num_edges x num_embeddings

    """
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=dev) * -(np.log(10000.0) / num_embeddings))
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def _orientations(X_ca):
    """ Compute forward and backward vectors per residue.

    Args
    ----
    X_ca : torch.FloatTensor
        Tensor specifying atomic backbone coordinates for CA atoms.
        Shape: num_res x 3

    Returns
    -------
    torch.FloatTensor
        Pairs of forward, backward vectors per residue.
        Shape: num_res x 2 x 3
    """
    # From https://github.com/drorlab/gvp-pytorch
    forward = _normalize(X_ca[1:] - X_ca[:-1])
    backward = _normalize(X_ca[:-1] - X_ca[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X):
    """ Compute vectors pointing in the approximate direction of the sidechain.

    Args
    ----
    X : torch.FloatTensor
        Tensor specifying atomic backbone coordinates.
        Shape: num_res x 4 x 3

    Returns
    -------
    vec : torch.FloatTensor
        Sidechain vectors.
        Shape: num_res x 3
    """
    # From https://github.com/drorlab/gvp-pytorch
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def _jing_featurize(protein, dev='cpu'):
    """ Featurize individual proteins for use in torch_geometric Data objects,
    as done in https://github.com/drorlab/gvp-pytorch

    Args
    ----
    protein : dict
        Dictionary of protein features

        - :code:`name` - PDB ID of the protein
        - :code:`coords` - list of dicts specifying backbone atom coordinates
        in the format of that outputted by :code:`parseCoords.py`
        - :code:`seq` - protein sequence
        - :code:`chain_idx` - an integer per residue such that each unique integer represents a unique chain

    Returns
    -------
    torch_geometric.data.Data
        Data object containing
        - :code:`x` - CA atomic coordinates
        - :code:`seq` - sequence of protein
        - :code:`name` - PDB ID of protein
        - :code:`node_s` - Node scalar features
        - :code:`node_v` - Node vector features
        - :code:`edge_s` - Edge scalar features
        - :code:`edge_v` - Edge vector features
        - :code:`edge_index` - Sparse representation of edge
        - :code:`mask` - Residue mask specifying residues with incomplete coordinate sets
    """
    name = protein['name']
    with torch.no_grad():
        coords = torch.as_tensor(protein['coords'], device=dev, dtype=torch.float32)
        seq = torch.as_tensor(protein['seq'], device=dev, dtype=torch.long)

        mask = torch.isfinite(coords.sum(dim=(1, 2)))
        coords[~mask] = np.inf

        X_ca = coords[:, 1]
        edge_index = torch_cluster.knn_graph(X_ca, k=30, loop=True)  # TODO: make param

        pos_embeddings = _positional_embeddings(edge_index)
        # generate mask for interchain interactions
        pos_chain = (protein['chain_idx'][edge_index.view(-1)]).view(2, -1)
        pos_mask = (pos_chain[0] != pos_chain[1])
        # zero out all interchain positional embeddings
        pos_embeddings = pos_mask.unsqueeze(-1) * pos_embeddings

        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=16, device=dev)  # TODO: make param

        dihedrals = _dihedrals(coords)
        orientations = _orientations(X_ca)
        sidechains = _sidechains(coords)

        node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)

        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))

    data = torch_geometric.data.Data(x=X_ca,
                                     seq=seq,
                                     name=name,
                                     node_s=node_s,
                                     node_v=node_v,
                                     edge_s=edge_s,
                                     edge_v=edge_v,
                                     edge_index=edge_index,
                                     mask=mask)
    return data


# Ingraham featurization functions


def _ingraham_featurize(batch, device="cpu"):
    """ Pack and pad coords in batch into torch tensors
    as done in https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    batch : list of dict
        list of protein backbone coordinate dictionaries,
        in the format of that outputted by :code:`parseCoords.py`
    device : str
        device to place torch tensors

    Returns
    -------
    X : torch.Tensor
        Batched coordinates tensor
    mask : torch.Tensor
        Mask for X
    lengths : np.ndarray
        Array of lengths of batched proteins
    """
    B = len(batch)
    lengths = np.array([b.shape[0] for b in batch], dtype=np.int32)
    l_max = max(lengths)
    X = np.zeros([B, l_max, 4, 3])

    # Build the batch
    for i, x in enumerate(batch):
        l = x.shape[0]
        x_pad = np.pad(x, [[0, l_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan, ))
        X[i, :, :, :] = x_pad

    # Mask
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    return X, mask, lengths

def _quaternions(R, eps=1e-10):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [...,3,3]
        Q [...,4]
    """
    def _R(i, j):
        return R[..., i, j]

    # Simple Wikipedia version
    # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(
        torch.abs(1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)) + eps)
    signs = torch.sign(torch.stack([_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q


def _orientations_coarse(X, edge_index, eps=1e-6):
    # Pair features

    # Shifted slices of unit vectors
    dX = X[1:, :] - X[:-1, :]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2, :]
    u_1 = U[1:-1, :]
    u_0 = U[2:, :]
    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Bond angle calculation
    cosA = -(u_1 * u_0).sum(-1)
    cosA = torch.clamp(cosA, -1 + eps, 1 - eps)
    A = torch.acos(cosA)
    # Angle between normals
    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
    # Backbone features
    AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), -1)
    AD_features = F.pad(AD_features, (0, 0, 1, 2), 'constant', 0)

    # Build relative orientations
    o_1 = F.normalize(u_2 - u_1, dim=-1)
    O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
    O = O.view(list(O.shape[:1]) + [9])
    O = F.pad(O, (0, 0, 1, 2), 'constant', 0)

    # DEBUG: Viz [dense] pairwise orientations
    # O = O.view(list(O.shape[:2]) + [3,3])
    # dX = X.unsqueeze(2) - X.unsqueeze(1)
    # dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
    # dU = dU / torch.norm(dU, dim=-1, keepdim=True)
    # dU = (dU + 1.) / 2.
    # plt.imshow(dU.data.numpy()[0])
    # plt.show()
    # print(dX.size(), O.size(), dU.size())
    # exit(0)

    O_pairs = O[edge_index]
    X_pairs = X[edge_index]

    # Re-view as rotation matrices
    O_pairs = O_pairs.view(list(O_pairs.shape[:-1]) + [3,3])

    # Rotate into local reference frames
    dX = X_pairs[0] - X_pairs[1]
    dU = torch.matmul(O_pairs[1], dX.unsqueeze(-1)).squeeze(-1)
    dU = F.normalize(dU, dim=-1)
    R = torch.matmul(O_pairs[1].transpose(-1, -2), O_pairs[0])
    Q = _quaternions(R)

    # Orientation features
    O_features = torch.cat((dU, Q), dim=-1)

    # DEBUG: Viz pairwise orientations
    # IMG = Q[:,:,:,:3]
    # # IMG = dU
    # dU_full = torch.zeros(X.shape[0], X.shape[1], X.shape[1], 3).scatter(
    #     2, E_idx.unsqueeze(-1).expand(-1,-1,-1,3), IMG
    # )
    # print(dU_full)
    # dU_full = (dU_full + 1.) / 2.
    # plt.imshow(dU_full.data.numpy()[0])
    # plt.show()
    # exit(0)
    # print(Q.sum(), dU.sum(), R.sum())
    return AD_features, O_features

def _ingraham_geometric_featurize(protein, dev='cpu'):
    """ Featurize individual proteins for use in torch_geometric Data objects,
    as done in https://github.com/drorlab/gvp-pytorch

    Args
    ----
    protein : dict
        Dictionary of protein features

        - :code:`name` - PDB ID of the protein
        - :code:`coords` - list of dicts specifying backbone atom coordinates
        in the format of that outputted by :code:`parseCoords.py`
        - :code:`seq` - protein sequence
        - :code:`chain_idx` - an integer per residue such that each unique integer represents a unique chain

    Returns
    -------
    torch_geometric.data.Data
        Data object containing
        - :code:`x` - CA atomic coordinates
        - :code:`seq` - sequence of protein
        - :code:`name` - PDB ID of protein
        - :code:`node_s` - Node scalar features
        - :code:`node_v` - Node vector features
        - :code:`edge_s` - Edge scalar features
        - :code:`edge_v` - Edge vector features
        - :code:`edge_index` - Sparse representation of edge
        - :code:`mask` - Residue mask specifying residues with incomplete coordinate sets
    """
    name = protein['name']
    with torch.no_grad():
        coords = torch.as_tensor(protein['coords'], device=dev, dtype=torch.float32)
        seq = torch.as_tensor(protein['seq'], device=dev, dtype=torch.long)

        mask = torch.isfinite(coords.sum(dim=(1, 2)))
        coords[~mask] = np.inf

        X_ca = coords[:, 1]
        edge_index = torch_cluster.knn_graph(X_ca, k=30, loop=True)  # TODO: make param

        pos_embeddings = _positional_embeddings(edge_index)
        # generate mask for interchain interactions
        pos_chain = (protein['chain_idx'][edge_index.view(-1)]).view(2, -1)
        pos_mask = (pos_chain[0] != pos_chain[1])
        # zero out all interchain positional embeddings
        pos_embeddings = pos_mask.unsqueeze(-1) * pos_embeddings

        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=16, device=dev)  # TODO: make param

        dihedrals = _dihedrals(coords)
        _, orientations = _orientations_coarse(X_ca, edge_index)

        node_features = dihedrals
        edge_features = torch.cat([pos_embeddings, rbf, orientations], dim=-1)

        node_features, edge_features, = map(torch.nan_to_num, (node_features, edge_features))

    data = torch_geometric.data.Data(x=X_ca,
                                     seq=seq,
                                     name=name,
                                     node_features=node_features,
                                     edge_features=edge_features,
                                     edge_index=edge_index,
                                     mask=mask)
    return data


# Batching functions


def convert(tensor):
    """Converts given tensor from numpy to pytorch."""
    return torch.from_numpy(tensor)


def _package(batch, k_neighbors=30):
    """Package the given datapoints into tensors based on provided indices.

    Tensors are extracted from the data and padded. Coordinates are featurized
    and the length of Coords and chain IDs are added to the data.

    Args
    ----
    batch : list dicts
        The feature dictionaries for each datapoint to package.

    Returns
    -------
    dict
        Collection of batched features required for running Coordinator. This contains:

        - :code:`seq_lens` - lengths of the target sequences

        - :code:`X` - coordinates

        - :code:`x_mask` - mask for the target structure

        - :code:`seqs` - the target sequences

        - :code:`ids` - the PDB ids

        - :code:`chain_idx` - the chain IDs
    """
    # wrap up all the tensors with proper padding and masks
    seq_lens, coords = [], []
    seqs = []
    ids = []
    chain_lens = []
    gvp_data = []
    geometric_data = []

    sortcery_seqs = []
    sortcery_nrgs = []


    for _, data in enumerate(batch):
        # have to transpose these two because then we can use pad_sequence for padding
        seq_lens.append(data['seq_len'])
        coords.append(data['coords'])
        seqs.append(convert(data['sequence']))
        ids.append(data['pdb'])
        chain_lens.append(data['chain_lens'])

        if 'sortcery_seqs' in data:
            assert len(batch) == 1, "batch_size for SORTCERY fine-tuning should be set to 1"
            sortcery_seqs = convert(data['sortcery_seqs']).unsqueeze(0)
        if 'sortcery_nrgs' in data:
            sortcery_nrgs = convert(data['sortcery_nrgs']).unsqueeze(0)

        chain_idx = []
        for i, c_len in enumerate(data['chain_lens']):
            chain_idx.append(torch.ones(c_len) * i)
        chain_idx = torch.cat(chain_idx, dim=0)
        gvp_data.append(
            _jing_featurize({
                'name': data['pdb'],
                'coords': data['coords'],
                'seq': data['sequence'],
                'chain_idx': chain_idx
            }))
        geometric_data.append(
            _ingraham_geometric_featurize({
                'name': data['pdb'],
                'coords': data['coords'],
                'seq': data['sequence'],
                'chain_idx': chain_idx
            }))
        


    # we can pad these using standard pad_sequence
    seqs = pad_sequence(seqs, batch_first=True)

    # featurize coordinates same way as ingraham et al
    # featurize coordinates same way as ingraham et al
    X, x_mask, _ = _ingraham_featurize(coords)


    # pad with -1 so we can store term_lens in a tensor
    seq_lens = torch.tensor(seq_lens)

    # generate chain_idx from chain_lens
    chain_idx = []
    for c_lens in chain_lens:
        arrs = []
        for i, chain_len in enumerate(c_lens):
            arrs.append(torch.ones(chain_len) * i)
        chain_idx.append(torch.cat(arrs, dim=-1))
    chain_idx = pad_sequence(chain_idx, batch_first=True)

    return {
        'seq_lens': seq_lens,
        'X': X,
        'x_mask': x_mask,
        'seqs': seqs,
        'ids': ids,
        'chain_idx': chain_idx,
        'gvp_data': gvp_data,
        'sortcery_seqs': sortcery_seqs,
        'sortcery_nrgs': sortcery_nrgs,
        'geometric_data': geometric_data,
    }


# Non-lazy data loading functions

def load_feature_file(in_folder, pdb_id, min_protein_len=30, k_neighbors=30):
    """Load the data specified in the proper .features file and return them.
    If the read sequence length is less than :code:`min_protein_len`, instead return None.

    Args
    ----
    in_folder : str
        folder to find TERM file.
    pdb_id : str
        PDB ID to load.
    min_protein_len : int
        minimum cutoff for loading TERM file.
    k_neighbors : int
        number of neighbors for knn graph

    Returns
    -------
    data : dict
        Data from TERM file (as dict)
    total_term_len : int
        Sum of lengths of all TERMs
    seq_len : int
        Length of protein sequence
    """
    path = f"{in_folder}/{pdb_id}/{pdb_id}.features"
    if not os.path.exists(path):
        print(f"{pdb_id} does not have feature file. Skipping.", flush=True)
        return None
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
        if len(data['sequence'].shape) > 1 and data['coords'].shape[0] != data['sequence'].shape[1]:
            print(f"{pdb_id} has structure/sequence consistency problems: struct len {data['coords'].shape[0]}, seq len: {data['sequence'].shape[1]}!", flush=True)
            return None
        seq_len = data['seq_len']
        if seq_len < min_protein_len:
            print(f"{pdb_id} has length problems: seq_len {seq_len}, min_protein_len: {min_protein_len}", flush=True)
            return None
    output = {
        'pdb': pdb_id,
        'coords': data['coords'],
        'res_info': None,
        'sequence': data['sequence'],
        'seq_len': seq_len,
        'chain_lens': data['chain_lens'],
    }
    return output, seq_len

def load_file(in_folder, pdb_id, min_protein_len=30, k_neighbors=30):
    """Load the data specified in the proper .pdb file
    If the read sequence length is less than :code:`min_protein_len`, instead return None.

    Args
    ----
    in_folder : str
        folder to find Coord file.
    pdb_id : str
        PDB ID to load.
    min_protein_len : int
        minimum cutoff for loading Coord file.
    k_neighbors : int
        number of neighbors for knn graph

    Returns
    -------
    data : dict
        Data from Coord file (as dict)
    seq_len : int
        Length of protein sequence
    """
    in_file = os.path.join(in_folder, pdb_id + '.pdb')
    if not os.path.exists(in_file):
        in_file = os.path.join(in_folder, pdb_id, pdb_id + '.red.pdb')
        if not os.path.exists(in_file):
            return load_feature_file(in_folder, pdb_id)
    try:
        coords, seq, res_info = parseCoords(in_file, save=False)
    except Exception as e:
        print(f"Unknown atom in pdb {pdb_id}. Skipping"+'\n')
        print(e)
        return None
    if len(seq) < min_protein_len:
        print(f"{pdb_id} is too short: {len(seq)} < {min_protein_len}")
        return None
    if len(coords) == 1:
        chain = next(iter(coords.keys()))
        coords_tensor = coords[chain]
    else:
        chains = sorted(coords.keys())
        coords_tensor = np.vstack([coords[c] for c in chains])

    chain_lens = [len(coords[c]) for c in sorted(coords.keys())]
    int_seq = np.array(seq_to_ints(seq))

    output = {
        'pdb': pdb_id,
        'coords': coords_tensor,
        'res_info': res_info,
        'sequence': int_seq,
        'seq_len': len(seq),
        'chain_lens': chain_lens,
    }

    return output, len(seq)


class CoordDataset(Dataset):
    """Coordinate Dataset that loads all feature files into a Pytorch Dataset-like structure.

    Attributes
    ----
    dataset : list
        list of tuples containing features and sequence length
    shuffle_idx : list
        array of indices for the dataset, for shuffling
    """
    def __init__(self, in_folder, pdb_ids=None, min_protein_len=30, k_neighbors=30, num_processes=32):
        """
        Initializes current Coord dataset by reading in feature files.

        Reads in all feature files from the given directory, using multiprocessing
        with the provided number of processes. Stores the features and the sequence 
        length as a tuple representing the data. Can read from PDB ids or
        file paths directly. Uses the given protein length as a cutoff.

        Args
        ----
        in_folder : str
            path to directory containing clean pdb files in `.pdb` or `.red.pdb` format
        pdb_ids: list, optional
            list of pdbs from `in_folder` to include in the dataset
        min_protein_len: int, default=30
            minimum length of a protein in the dataset
        k_neighbors : int
            number of neighbors for knn graph
        num_processes: int, default=32
            number of processes to use during dataloading
        """
        self.dataset = []
        with mp.Pool(num_processes) as pool:

            if pdb_ids:
                # progress = tqdm(total=len(pdb_ids))

                def update_progress(res):
                    del res
                    # progress.update(1)

                res_list = [
                    pool.apply_async(load_file, (in_folder, id),
                                     kwds={"min_protein_len": min_protein_len, "k_neighbors": k_neighbors}
                                     ) for id in pdb_ids
                ]
                pool.close()
                pool.join()
                # progress.close()
                for res in res_list:
                    data = res.get()
                    if data is not None:
                        features, seq_len = data
                        self.dataset.append((features, seq_len))
            else:
                print("Loading pdb file paths")
                filelist = list(glob.glob(f'{in_folder}/*.pdb'))
                ext = '.pdb'
                if len(filelist) == 0:
                    filelist = list(glob.glob(f'{in_folder}/*/*.red.pdb'))
                    ext = '.red.pdb'
                assert len(filelist) > 0,f"No pdb files found: {in_folder}"+'\n'
                progress = tqdm(total=len(filelist))

                def update_progress(res):
                    del res
                    progress.update(1)

                # get pdb_ids
                pdb_ids = [os.path.basename(path)[:-len(ext)] for path in filelist]

                res_list = [
                    pool.apply_async(load_file, (in_folder, id),
                                     kwds={"min_protein_len": min_protein_len, "k_neighbors": k_neighbors},
                                     callback=update_progress) for id in pdb_ids
                ]
                pool.close()
                pool.join()
                progress.close()
                for res in res_list:
                    data = res.get()
                    if data is not None:
                        features, seq_len = data
                        self.dataset.append((features, seq_len))

            self.shuffle_idx = np.arange(len(self.dataset))

    def shuffle(self):
        """Shuffle the current dataset."""
        np.random.shuffle(self.shuffle_idx)

    def __len__(self):
        """Returns length of the given dataset.

        Returns
        -------
        int
            length of dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """Extract a given item with provided index.

        Args
        ----
        idx : int
            Index of item to return.
        Returns
        ----
        data : dict
            Data from Coord file (as dict)
        seq_len : int
            Length of protein sequence
        """
        data_idx = self.shuffle_idx[idx]
        if isinstance(data_idx, list):
            return [self.dataset[i] for i in data_idx]
        return self.dataset[data_idx]


class CoordBatchSampler(Sampler):
    """BatchSampler/Dataloader helper class for Coord data using CoordDataset.

    Attributes
    ----
    size: int
        Length of the dataset
    dataset: List
        List of features from Coord dataset
    seq_lengths: List
        List of sequence lengths from the given dataset
    batch_size : int or None, default=4
        Size of batches created. If variable sized batches are desired, set to None.
    sort_data : bool, default=False
        Create deterministic batches by sorting the data according to the
        specified length metric and creating batches from the sorted data.
        Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
    shuffle : bool, default=True
        Shuffle the data completely before creating batches.
        Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
    semi_shuffle : bool, default=False
        Sort the data according to the specified length metric,
        then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
        Within each partition perform a complete shuffle. The upside is that
        batching with similar lengths reduces padding making for more efficient computation,
        but the downside is that it does a less complete shuffle.
    semi_shuffle_cluster_size : int, default=500
        Size of partition to use when :code:`semi_shuffle=True`.
    batch_shuffle : bool, default=True
        If set to :code:`True`, shuffle samples within a batch.
    drop_last : bool, default=False
        If set to :code:`True`, drop the last samples if they don't form a complete batch.
    max_seq_tokens : int, default 8000
        When :code:`batch_size=None, max_seq_tokens>0`,
        batch by fitting as many datapoints as possible with the total number of
        sequence residues included below `max_seq_tokens`.
    """
    def __init__(self,
                 dataset,
                 batch_size=4,
                 sort_data=False,
                 shuffle=True,
                 semi_shuffle=False,
                 semi_shuffle_cluster_size=500,
                 batch_shuffle=True,
                 drop_last=False,
                 max_seq_tokens=8000):
        """
        Reads in and processes a given dataset.

        Given the provided dataset, load all the data. Then cluster the data using
        the provided method, either shuffled or sorted and then shuffled.

        Args
        ----
        dataset : CoordDataset
            Dataset to batch.
        batch_size : int or None, default=4
            Size of batches created. If variable sized batches are desired, set to None.
        sort_data : bool, default=False
            Create deterministic batches by sorting the data according to the
            specified length metric and creating batches from the sorted data.
            Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
        shuffle : bool, default=True
            Shuffle the data completely before creating batches.
            Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
        semi_shuffle : bool, default=False
            Sort the data according to the specified length metric,
            then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
            Within each partition perform a complete shuffle. The upside is that
            batching with similar lengths reduces padding making for more efficient computation,
            but the downside is that it does a less complete shuffle.
        semi_shuffle_cluster_size : int, default=500
            Size of partition to use when :code:`semi_shuffle=True`.
        batch_shuffle : bool, default=True
            If set to :code:`True`, shuffle samples within a batch.
        drop_last : bool, default=False
            If set to :code:`True`, drop the last samples if they don't form a complete batch.
        max_seq_tokens : int or None, default=None
            When :code:`batch_size=None, max_seq_tokens>0`,
            batch by fitting as many datapoints as possible with the total number of
            sequence residues included below `max_seq_tokens`.
        """
        super().__init__(dataset)
        self.size = len(dataset)
        self.dataset, self.seq_lengths = zip(*dataset)
        self.shuffle = shuffle
        self.sort_data = sort_data
        self.batch_shuffle = batch_shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.max_seq_tokens = max_seq_tokens
        self.semi_shuffle = semi_shuffle
        self.semi_shuffle_cluster_size = semi_shuffle_cluster_size

        assert not (shuffle and semi_shuffle), "Lazy Dataloader shuffle and semi shuffle cannot both be set"

        # initialize clusters
        self._cluster()

    def _cluster(self):
        """ Shuffle data and make clusters of indices corresponding to batches of data.

        This method speeds up training by sorting data points with similar Coord lengths
        together, if :code:`sort_data` or :code:`semi_shuffle` are on. Under `sort_data`,
        the data is sorted by length. Under `semi_shuffle`, the data is broken up
        into clusters based on length and shuffled within the clusters. Otherwise,
        it is randomly shuffled. Data is then loaded into batches based on the number
        of proteins that will fit into the GPU without overloading it, based on
        :code:`max_seq_tokens`.
        """

        # if we sort data, use sorted indexes instead
        if self.sort_data:
            idx_list = np.argsort(self.seq_lengths)
        elif self.semi_shuffle:
            # trying to speed up training
            # by shuffling points with similar term res together
            idx_list = np.argsort(self.seq_lengths)
            shuffle_borders = []

            # break up datapoints into large clusters
            border = 0
            while border < len(self.seq_lengths):
                shuffle_borders.append(border)
                border += self.semi_shuffle_cluster_size

            # shuffle datapoints within clusters
            last_cluster_idx = len(shuffle_borders) - 1
            for cluster_idx in range(last_cluster_idx + 1):
                start = shuffle_borders[cluster_idx]
                if cluster_idx < last_cluster_idx:
                    end = shuffle_borders[cluster_idx + 1]
                    np.random.shuffle(idx_list[start:end])
                else:
                    np.random.shuffle(idx_list[start:])

        else:
            idx_list = list(range(len(self.dataset)))
            np.random.shuffle(idx_list)

        # Cluster into batches of similar sizes
        clusters, batch = [], []

        # if batch_size is None, fit as many proteins we can into a batch
        # without overloading the GPU
        if self.batch_size is None:
            current_batch_lens = []
            total_data_len = 0
            for count, idx in enumerate(idx_list):
                current_batch_lens.append(self.seq_lengths[idx])
                total_data_len = max(current_batch_lens) * len(current_batch_lens)
                if count != 0 and total_data_len > self.max_seq_tokens:
                    clusters.append(batch)
                    batch = [idx]
                    current_batch_lens = [self.seq_lengths[idx]]
                else:
                    batch.append(idx)

        else:  # used fixed batch size
            for count, idx in enumerate(idx_list):
                if count != 0 and count % self.batch_size == 0:
                    clusters.append(batch)
                    batch = [idx]
                else:
                    batch.append(idx)

        if len(batch) > 0 and not self.drop_last:
            clusters.append(batch)
        self.clusters = clusters

    def package(self, b_idx):
        """Package the given datapoints into tensors based on provided indices.

        Tensors are extracted from the data and padded. Coordinates are featurized
        and the length of Coords and chain IDs are added to the data.

        Args
        ----
        b_idx : list of tuples (dicts, int, int)
            The feature dictionaries, the sum of the lengths of all Coords, and the sum of all sequence lengths
            for each datapoint to package.

        Returns
        -------
        dict
            Collection of batched features required for running Coordinator. This contains:

            - :code:`seq_lens` - lengths of the target sequences

            - :code:`X` - coordinates

            - :code:`x_mask` - mask for the target structure

            - :code:`seqs` - the target sequences

            - :code:`ids` - the PDB ids

            - :code:`chain_idx` - the chain IDs
        """
        return _package([b[0] for b in b_idx])

    def __len__(self):
        """Returns length of dataset, i.e. number of batches.

        Returns
        -------
        int
            length of dataset.
        """
        return len(self.clusters)

    def __iter__(self):
        """Allows iteration over dataset."""
        if self.shuffle or self.semi_shuffle:
            self._cluster()
            np.random.shuffle(self.clusters)
        for batch in self.clusters:
            yield batch


# needs to be outside of object for pickling reasons (?)
def read_lens(in_folder, pdb_id, min_protein_len=30):
    """ Reads the lengths specified in the proper .pdb or .red.pdb file and return them.

    If the read sequence length is less than :code:`min_protein_len`, instead return None.

    Args
    ----
    in_folder : str
        folder to find Coord file.
    pdb_id : str
        PDB ID to load.
    min_protein_len : int
        minimum cutoff for loading Coord file.
    Returns
    -------
    pdb_id : str
        PDB ID that was loaded
    seq_len : int
        sequence length of file, or None if sequence length is less than :code:`min_protein_len`
    """
    path = f"{in_folder}/{pdb_id}.pdb"
    if not os.path.exists(path):
        path = f"{in_folder}/{pdb_id}/{pdb_id}.red.pdb"
    # pylint: disable=unspecified-encoding
    with open(path, 'rt') as fp:
        try:
            _, seq, _ = parseCoords(path, save=False)
        except Exception as e:
            print(f"Unknown atom in pdb {pdb_id}. Skipping")
            print(e)
            return None
        seq_len = len(seq)
        if seq_len < min_protein_len:
            return None
        _ = seq_to_ints(seq)
        
    return pdb_id, seq_len


class CoordLazyDataset(Dataset):
    """Coord Dataset that loads all feature files into a Pytorch Dataset-like structure.

    Unlike CoordDataset, this just loads feature filenames, not actual features.

    Attributes
    ----
    dataset : list
        list of tuples containing feature filenames, Coord length, and sequence length
    shuffle_idx : list
        array of indices for the dataset, for shuffling
    """
    def __init__(self, in_folder, pdb_ids=None, min_protein_len=30, num_processes=32):
        """
        Initializes current Coord dataset by reading in feature files.

        Reads in all feature files from the given directory, using multiprocessing
        with the provided number of processes. Stores the feature filenames, the Coord length,
        and the sequence length as a tuple representing the data. Can read from PDB ids or
        file paths directly. Uses the given protein length as a cutoff.

        Args
        ----
        in_folder : str
            path to directory containing clean pdb files in `.pdb` or `.red.pdb` format
        pdb_ids: list, optional
            list of pdbs from `in_folder` to include in the dataset
        min_protein_len: int, default=30
            minimum length of a protein in the dataset
        num_processes: int, default=32
            number of processes to use during dataloading
        """
        self.dataset = []

        with mp.Pool(num_processes) as pool:

            if pdb_ids:
                print("Loading feature file paths")
                progress = tqdm(total=len(pdb_ids))

                def update_progress(res):
                    del res
                    progress.update(1)

                res_list = [
                    pool.apply_async(read_lens, (in_folder, pdb_id),
                                     kwds={"min_protein_len": min_protein_len},
                                     callback=update_progress) for pdb_id in pdb_ids
                ]
                pool.close()
                pool.join()
                progress.close()
                for res in res_list:
                    data = res.get()
                    if data is not None:
                        pdb_id, seq_len = data
                        filename = f"{in_folder}/{pdb_id}.pdb"
                        if not os.path.exists(filename):
                            filename = f"{in_folder}/{pdb_id}/{pdb_id}.red.pdb"
                        self.dataset.append((os.path.abspath(filename), seq_len))
            else:
                print("Loading feature file paths")
                filelist = list(glob.glob(f'{in_folder}/*/*.red.pdb'))
                ext = '.pdb'
                if len(filelist) == 0:
                    filelist = list(glob.glob(f'{in_folder}/*/*.red.pdb'))
                    ext = '.red.pdb'
                progress = tqdm(total=len(filelist))

                def update_progress(res):
                    del res
                    progress.update(1)

                # get pdb_ids
                pdb_ids = [os.path.basename(path)[:-len(ext)] for path in filelist]

                res_list = [
                    pool.apply_async(read_lens, (in_folder, pdb_id),
                                     kwds={"min_protein_len": min_protein_len},
                                     callback=update_progress) for pdb_id in pdb_ids
                ]
                pool.close()
                pool.join()
                progress.close()
                for res in res_list:
                    data = res.get()
                    if data is not None:
                        pdb_id, seq_len = data
                        filename = f"{in_folder}/{pdb_id}.pdb"
                        if not os.path.exists(filename):
                            filename = f"{in_folder}/{pdb_id}/{pdb_id}.red.pdb"
                        self.dataset.append((os.path.abspath(filename), seq_len))

        self.shuffle_idx = np.arange(len(self.dataset))

    def shuffle(self):
        """Shuffle the dataset"""
        np.random.shuffle(self.shuffle_idx)

    def __len__(self):
        """Returns length of the given dataset.

        Returns
        -------
        int
            length of dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """Extract a given item with provided index.

        Args
        ----
        idx : int
            Index of item to return.
        Returns
        ----
        data : dict
            Data from Coord file (as dict)
        seq_len : int
            Length of protein sequence
        """
        data_idx = self.shuffle_idx[idx]
        if isinstance(data_idx, list):
            return [self.dataset[i] for i in data_idx]
        return self.dataset[data_idx]


class CoordLazyBatchSampler(Sampler):
    """BatchSampler/Dataloader helper class for Coord data using CoordLazyDataset.

    Attributes
    ----------
    dataset : CoordLazyDataset
        Dataset to batch.
    size : int
        Length of dataset
    batch_size : int or None, default=4
        Size of batches created. If variable sized batches are desired, set to None.
    sort_data : bool, default=False
        Create deterministic batches by sorting the data according to the
        specified length metric and creating batches from the sorted data.
        Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
    shuffle : bool, default=True
        Shuffle the data completely before creating batches.
        Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
    semi_shuffle : bool, default=False
        Sort the data according to the specified length metric,
        then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
        Within each partition perform a complete shuffle. The upside is that
        batching with similar lengths reduces padding making for more efficient computation,
        but the downside is that it does a less complete shuffle.
    semi_shuffle_cluster_size : int, default=500
        Size of partition to use when :code:`semi_shuffle=True`.
    batch_shuffle : bool, default=True
        If set to :code:`True`, shuffle samples within a batch.
    drop_last : bool, default=False
        If set to :code:`True`, drop the last samples if they don't form a complete batch.
    max_seq_tokens : int, default=8000
        When :code:`batch_size=None, max_term_res=None, max_seq_tokens>0`,
        batch by fitting as many datapoints as possible with the total number of
        sequence residues included below `max_seq_tokens`.
    """
    def __init__(self,
                 dataset,
                 batch_size=4,
                 sort_data=False,
                 shuffle=True,
                 semi_shuffle=False,
                 semi_shuffle_cluster_size=500,
                 batch_shuffle=True,
                 drop_last=False,
                 max_seq_tokens=8000):
        """
        Reads in and processes a given dataset.

        Given the provided dataset, load all the data. Then cluster the data using
        the provided method, either shuffled or sorted and then shuffled.

        Args
        ----
        dataset : CoordLazyDataset
            Dataset to batch.
        batch_size : int or None, default=4
            Size of batches created. If variable sized batches are desired, set to None.
        sort_data : bool, default=False
            Create deterministic batches by sorting the data according to the
            specified length metric and creating batches from the sorted data.
            Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
        shuffle : bool, default=True
            Shuffle the data completely before creating batches.
            Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
        semi_shuffle : bool, default=False
            Sort the data according to the specified length metric,
            then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
            Within each partition perform a complete shuffle. The upside is that
            batching with similar lengths reduces padding making for more efficient computation,
            but the downside is that it does a less complete shuffle.
        semi_shuffle_cluster_size : int, default=500
            Size of partition to use when :code:`semi_shuffle=True`.
        batch_shuffle : bool, default=True
            If set to :code:`True`, shuffle samples within a batch.
        drop_last : bool, default=False
            If set to :code:`True`, drop the last samples if they don't form a complete batch.
        max_seq_tokens : int or None, default=None
            When :code:`batch_size=None, max_term_res=None, max_seq_tokens>0`,
            batch by fitting as many datapoints as possible with the total number of
            sequence residues included below `max_seq_tokens`.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.size = len(dataset)
        self.filepaths, self.seq_lengths = zip(*dataset)
        self.shuffle = shuffle
        self.sort_data = sort_data
        self.batch_shuffle = batch_shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.max_seq_tokens = max_seq_tokens
        self.semi_shuffle = semi_shuffle
        self.semi_shuffle_cluster_size = semi_shuffle_cluster_size

        assert not (shuffle and semi_shuffle), "Lazy Dataloader shuffle and semi shuffle cannot both be set"

        # initialize clusters
        self._cluster()

    def _cluster(self):
        """ Shuffle data and make clusters of indices corresponding to batches of data.

        This method speeds up training by sorting data points with similar Coord lengths
        together, if :code:`sort_data` or :code:`semi_shuffle` are on. Under `sort_data`,
        the data is sorted by length. Under `semi_shuffle`, the data is broken up
        into clusters based on length and shuffled within the clusters. Otherwise,
        it is randomly shuffled. Data is then loaded into batches based on the number
        of proteins that will fit into the GPU without overloading it, based on
        :code:`max_term_res` or :code:`max_seq_tokens`.
        """

        # if we sort data, use sorted indexes instead
        if self.sort_data:
            idx_list = np.argsort(self.seq_lengths)
        elif self.semi_shuffle:
            # trying to speed up training
            # by shuffling points with similar term res together
            idx_list = np.argsort(self.seq_lengths)
            shuffle_borders = []

            # break up datapoints into large clusters
            border = 0
            while border < len(self.seq_lengths):
                shuffle_borders.append(border)
                border += self.semi_shuffle_cluster_size

            # shuffle datapoints within clusters
            last_cluster_idx = len(shuffle_borders) - 1
            for cluster_idx in range(last_cluster_idx + 1):
                start = shuffle_borders[cluster_idx]
                if cluster_idx < last_cluster_idx:
                    end = shuffle_borders[cluster_idx + 1]
                    np.random.shuffle(idx_list[start:end])
                else:
                    np.random.shuffle(idx_list[start:])

        else:
            idx_list = list(range(len(self.dataset)))
            np.random.shuffle(idx_list)

        # Cluster into batches of similar sizes
        clusters, batch = [], []

        # if batch_size is None, fit as many proteins we can into a batch
        # without overloading the GPU
        if self.batch_size is None:
            current_batch_lens = []
            total_data_len = 0
            for count, idx in enumerate(idx_list):
                current_batch_lens.append(self.seq_lengths[idx])
                total_data_len = max(current_batch_lens) * len(current_batch_lens)
                if count != 0 and total_data_len > self.max_seq_tokens:
                    clusters.append(batch)
                    batch = [idx]
                    current_batch_lens = [self.seq_lengths[idx]]
                else:
                    batch.append(idx)

        else:  # used fixed batch size
            for count, idx in enumerate(idx_list):
                if count != 0 and count % self.batch_size == 0:
                    clusters.append(batch)
                    batch = [idx]
                else:
                    batch.append(idx)

        if len(batch) > 0 and not self.drop_last:
            clusters.append(batch)
        self.clusters = clusters

    def package(self, b_idx):
        """Package the given datapoints into tensors based on provided indices.

        Tensors are extracted from the data and padded. Coordinates are featurized
        and the length of Coords and chain IDs are added to the data.

        Args
        ----
        b_idx : list of (str, int)
            The path to the `.pdb` or `.red.pdb` files, the sum of the lengths of all Coords, and the sum of all sequence lengths
            for each datapoint to package.

        Returns
        -------
        dict
            Collection of batched features required for running Coordinator. This contains:

            - :code:`seq_lens` - lengths of the target sequences

            - :code:`X` - coordinates

            - :code:`x_mask` - mask for the target structure

            - :code:`seqs` - the target sequences

            - :code:`ids` - the PDB ids

            - :code:`chain_idx` - the chain IDs
        """
        if self.batch_shuffle:
            b_idx_copy = b_idx[:]
            random.shuffle(b_idx_copy)
            b_idx = b_idx_copy

        # load the files specified by filepaths
        batch = []
        for data in b_idx:
            filepath = data[0]
            pdb_id = os.path.basename(filepath).split('.')[0]
            coords, seq, _ = parseCoords(filepath, save=False)
            if len(coords) == 1:
                chain = next(iter(coords.keys()))
                coords_tensor = coords[chain]
            else:
                chains = sorted(coords.keys())
                coords_tensor = np.vstack([coords[c] for c in chains])

            chain_lens = [len(coords[c]) for c in sorted(coords.keys())]
            int_seq = np.array(seq_to_ints(seq))
            output = {
                'pdb': pdb_id,
                'coords': coords_tensor,
                'sequence': int_seq,
                'seq_len': len(seq),
                'chain_lens': chain_lens
            }
            batch.append(output)

        # package batch
        packaged_batch = _package(batch)

        return packaged_batch

    def __len__(self):
        """Returns length of dataset, i.e. number of batches.

        Returns
        -------
        int
            length of dataset.
        """
        return len(self.clusters)

    def __iter__(self):
        """Allows iteration over dataset."""
        if self.shuffle or self.semi_shuffle:
            self._cluster()
            np.random.shuffle(self.clusters)
        for batch in self.clusters:
            yield batch
