""" Backbone featurization modules

This file contains modules which featurize the protein backbone graph via
its backbone coordinates. Adapted from https://github.com/jingraham/neurips19-graph-protein-design
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .utils import (extract_knn, gather_edges, gather_nodes,  per_node_to_all_comb, 
                    average_duplicates, concatenate_duplicates)
from torch.nn.utils.rnn import pad_sequence

# pylint: disable=no-member

class PositionalEncodings(nn.Module):
    """ Module to generate differential positional encodings for protein graph edges """
    def __init__(self, num_embeddings):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx):
        """ Generate directional differential positional encodings for edges

        Args
        ----
        E_idx : torch.LongTensor
            Protein kNN edge indices
            Shape: n_batches x seq_len x k

        Returns
        -------
        E : torch.Tensor
            Directional Diffential positional encodings for edges
            Shape: n_batches x seq_len x k x num_embeddings
        """
        dev = E_idx.device
        # i-j
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1)).to(dev)
        d = (E_idx.float() - ii).unsqueeze(-1)
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32) *
            -(np.log(10000.0) / self.num_embeddings)).to(dev)
        angles = d * frequency.view((1, 1, 1, -1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

class PositionalChainEncodings(nn.Module):
    """ Module to generate differential positional encodings for chain info for graph edges
        Adapted from https://github.com/dauparas/ProteinMPNN
    """
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalChainEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E

class ProteinFeatures(nn.Module):
    """ Protein backbone featurization based on Ingraham et al NeurIPS

    Attributes
    ----------
    embeddings : PositionalEncodings
        Module to generate differential positional embeddings for edges
    dropout : nn.Dropout
        Dropout module
    node_embeddings, edge_embeddings : nn.Linear
        Embedding layers for nodes and edges
    norm_nodes, norm_edges : nn.LayerNorm
        Normalization layers for node and edge features
    """
    def __init__(self,
                 edge_features,
                 node_features,
                 num_positional_embeddings=16,
                 num_rbf=16,
                 top_k=30,
                 features_type='full',
                 features_options='',
                 augment_eps=0.,
                 dropout=0.1,
                 chain_handle='',
                 zero_out_pos_embs=False, # if true, just zero out the pos embedding but keep it around
                ):
        """ Extract protein features """
        super().__init__()
        self.edge_features = edge_features
        if features_options.find('concatenate_short') > -1:
            self.edge_features = int(self.edge_features / 2)
        elif features_options.find('concatenate_long') > -1:
            self.edge_shrink = nn.Linear(self.edge_features * 2, self.edge_features, bias=False)
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        # Feature types
        self.features_type = features_type
        self.zero_out_pos_embs = zero_out_pos_embs
        self.feature_dimensions = {
            'coarse': (3, num_positional_embeddings + num_rbf + 7),
            'full': (6, num_positional_embeddings + num_rbf + 7),
            'dist': (6, num_positional_embeddings + num_rbf),
            'hbonds': (3, 2 * num_positional_embeddings),
        }

        # Positional encoding
        # this chain embeddings is unused?
        #self.chain_embeddings = PositionalChainEncodings(num_positional_embeddings)
        if num_positional_embeddings > 0:
            self.embeddings = PositionalEncodings(num_positional_embeddings)
        else:
            self.embeddings = None
        
        self.dropout = nn.Dropout(dropout)

        # Normalization and embedding
        node_in, edge_in = self.feature_dimensions[features_type]
        self.chain_handle = chain_handle
        self.features_options = features_options
        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = nn.LayerNorm(node_features)  # Normalize(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)  # Normalize(edge_features)
        print("ZERO OUT POS EMB", self.zero_out_pos_embs)

        if self.features_options.find('nonlinear') > -1:
            self.nodes_nonlinear = nn.ReLU(inplace=False)
            self.edges_nonlinear = nn.ReLU(inplace=False)

    def _dist(self, X, mask, eps=1E-6, chain_ids=None):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D, D_neighbors, E_idx = extract_knn(X, mask, eps, self.top_k, chain_ids=chain_ids)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        dev = D.device
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(dev)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _quaternions(self, R, eps=1e-10):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        def _R(i, j):
            return R[:, :, :, i, j]

        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(
            torch.abs(1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1) + eps))
        signs = torch.sign(torch.stack([_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        return Q

    def _contacts(self, D_neighbors, mask_neighbors, cutoff=8):
        """ Contacts """
        D_neighbors = D_neighbors.unsqueeze(-1)
        neighbor_C = mask_neighbors * (D_neighbors < cutoff).type(torch.float32)
        return neighbor_C

    def _hbonds(self, X, E_idx, mask_neighbors, eps=1E-3):
        """ Hydrogen bonds and contact map
        """
        X_atoms = dict(zip(['N', 'CA', 'C', 'O'], torch.unbind(X, 2)))

        # Virtual hydrogens
        X_atoms['C_prev'] = F.pad(X_atoms['C'][:, 1:, :], (0, 0, 0, 1), 'constant', 0)
        X_atoms['H'] = X_atoms['N'] + F.normalize(
            F.normalize(X_atoms['N'] - X_atoms['C_prev'], -1) + F.normalize(X_atoms['N'] - X_atoms['CA'], -1), -1)

        def _distance(X_a, X_b):
            return torch.norm(X_a[:, None, :, :] - X_b[:, :, None, :], dim=-1)

        def _inv_distance(X_a, X_b):
            return 1. / (_distance(X_a, X_b) + eps)

        # DSSP vacuum electrostatics model
        U = (0.084 * 332) * (_inv_distance(X_atoms['O'], X_atoms['N']) + _inv_distance(X_atoms['C'], X_atoms['H']) -
                             _inv_distance(X_atoms['O'], X_atoms['H']) - _inv_distance(X_atoms['C'], X_atoms['N']))

        HB = (U < -0.5).type(torch.float32)
        neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1), E_idx)
        return neighbor_HB

    def _orientations_coarse(self, X, E_idx, eps=1e-6, frame="residue"):
        # Pair features

        if frame == "neighbors":
            # Shifted slices of unit vectors
            dX = X[:, 1:, :] - X[:, :-1, :]
            U = F.normalize(dX, dim=-1)
            u_2 = U[:, :-2, :]
            u_1 = U[:, 1:-1, :]
            u_0 = U[:, 2:, :]
            # Backbone normals
            n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
            n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

            # Bond angle calculation
            cosA = -(u_1 * u_0).sum(-1)
            cosD = (n_2 * n_1).sum(-1)
            signD = (u_2 * n_1).sum(-1)
            
            # Build relative orientations
            o_1 = F.normalize(u_2 - u_1, dim=-1)
            O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
            O = O.view(list(O.shape[:2]) + [9])
            O = F.pad(O, (0, 0, 1, 2), 'constant', 0)
        elif frame == "residue":
            N_CA = X[:, :, 1, :] - X[:, :, 0, :]
            N_CA = F.normalize(N_CA, dim=-1)
            CA_C = X[:, :, 2, :] - X[:, :, 1, :]
            CA_C = F.normalize(CA_C, dim=-1)
            C_O = X[:, :, 3, :] - X[:, :, 2, :]
            C_O = F.normalize(C_O, dim=-1)
            
            n_1 = F.normalize(torch.cross(N_CA, CA_C), dim=-1)
            n_2 = F.normalize(torch.cross(CA_C, C_O), dim=-1)
            b = F.normalize((CA_C-N_CA), dim=-1)
            x = F.normalize(torch.cross(b, n_1), dim=-1)
            O = torch.stack((b, n_1, x), 2)
            O = O.view(list(O.shape[:2]) + [9])

            cosA = (N_CA * CA_C).sum(-1)
            
            cosD = (n_1 * n_2).sum(-1)
            signD = (C_O * n_1).sum(-1)
            X = X[:,:,1,:]
        else:
            print(f"frame {frame} not recognized.")
            return 0

        # Bond angle calculation
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(signD) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        AD_features = F.pad(AD_features, (0, 0, 1, 2), 'constant', 0)

        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU, Q), dim=-1)

        return AD_features, O_features

    def _dihedrals(self, X, mask, chain_dict=None, chain_handle='replace', eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1, 2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1) / 3), 3))

        # Get chain begin/end index info
        if chain_handle:
            chain_begin, chain_end, singles = chain_dict['begin'], chain_dict['end'], chain_dict['singles']
            if chain_handle == 'mask':
                D[:,:,0] *= chain_begin * singles
                D[:,:,1] *= chain_end * singles
                D[:,:,2] *= chain_end * singles
            elif chain_handle == 'replace':
                D[:,:,0] = torch.gather(D[:,:,0], 1, chain_begin) * singles
                D[:,:,1] = torch.gather(D[:,:,1], 1, chain_end) * singles
                D[:,:,2] = torch.gather(D[:,:,2], 1, chain_end) * singles

        # print(cosD.cpu().data.numpy().flatten())
        # print(omega.sum().cpu().data.numpy().flatten())

        # Bond angle calculation
        # A = torch.acos(-(u_1 * u_0).sum(-1))

        # DEBUG: Ramachandran plot
        # x = phi.cpu().data.numpy().flatten()
        # y = psi.cpu().data.numpy().flatten()
        # plt.scatter(x * 180 / np.pi, y * 180 / np.pi, s=1, marker='.')
        # plt.xlabel('phi')
        # plt.ylabel('psi')
        # plt.axis('square')
        # plt.grid()
        # plt.axis([-180,180,-180,180])
        # plt.show()

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

    def forward(self, X, mask, inds_reduce, inds_transpose, inds_duplicate, inds_singles):
        """ Featurize coordinates as an attributed graph

        Args
        ----
        X : torch.Tensor
            Backbone coordinates
            Shape: n_batch x seq_len x 4 x 3
        mask : torch.ByteTensor
            Mask for residues
            Shape: n_batch x seq_len

        Returns
        -------
        V : torch.Tensor
            Node embeddings
            Shape: n_batches x seq_len x n_hidden
        E : torch.Tensor
            Edge embeddings in kNN dense form
            Shape: n_batches x seq_len x k x n_hidden
        E_idx : torch.LongTensor
            Edge indices
            Shape: n_batches x seq_len x k x n_hidden
        """
        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        X_ca = X[:, :, 1, :]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask)

        # Pairwise features
        AD_features, O_features = self._orientations_coarse(X_ca, E_idx)
        RBF = self._rbf(D_neighbors)




        if self.features_type == 'coarse':
            # Coarse backbone features
            V = AD_features
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == 'hbonds':
            # Hydrogen bonds and contacts
            neighbor_HB = self._hbonds(X, E_idx, mask_neighbors)
            neighbor_C = self._contacts(D_neighbors, mask_neighbors)
            # Dropout
            neighbor_C = self.dropout(neighbor_C)
            neighbor_HB = self.dropout(neighbor_HB)
            # Pack
            V = mask.unsqueeze(-1) * torch.ones_like(AD_features)
            neighbor_C = neighbor_C.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            neighbor_HB = neighbor_HB.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            E = torch.cat((neighbor_C, neighbor_HB), -1)
        elif self.features_type == 'full':
            # Full backbone angles
            V = self._dihedrals(X)
            E = torch.cat((RBF, O_features), -1)
        elif self.features_type == 'dist':
            # Full backbone angles
            V = self._dihedrals(X)
            E = RBF
            
        # Pairwise embeddings
        if self.embeddings is not None:
            E_positional = self.embeddings(E_idx)
            if self.zero_out_pos_embs:
                E_positional = E_positional*0
            E = torch.cat((E_positional, E), -1)
        # Embed the nodes
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return V, E, E_idx


class IndexDiffEncoding(nn.Module):
    """ Module to generate differential positional encodings for multichain protein graph edges

    Similar to ProteinFeatures, but zeros out features between interchain interactions """
    def __init__(self, num_embeddings):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx, chain_idx):
        """ Generate directional differential positional encodings for edges

        Args
        ----
        E_idx : torch.LongTensor
            Protein kNN edge indices
            Shape: n_batches x seq_len x k
        chain_idx : torch.LongTensor
            Indices for residues such that each chain is assigned a unique integer
            and each residue in that chain is assigned that integer
            Shape: n_batches x seq_len

        Returns
        -------
        E : torch.Tensor
            Directional Diffential positional encodings for edges
            Shape: n_batches x seq_len x k x num_embeddings
        """
        dev = E_idx.device
        # i-j
        N_batch = E_idx.size(0)
        N_terms = E_idx.size(1)
        N_nodes = E_idx.size(2)
        N_neighbors = E_idx.size(3)
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1)).to(dev)
        d = (E_idx.float() - ii).unsqueeze(-1)

        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32) *
            -(np.log(10000.0) / self.num_embeddings)).to(dev)
        angles = d * frequency.view((1, 1, 1, -1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)

        # we zero out positional frequencies from inter-chain edges
        # the idea is, the concept of "sequence distance"
        # between two residues in different chains doesn't
        # make sense :P
        chain_idx_expand = chain_idx.view(N_batch, 1, -1, 1).expand((-1, N_terms, -1, N_neighbors))
        E_chain_idx = torch.gather(chain_idx_expand.to(dev), 2, E_idx)
        same_chain = (E_chain_idx == E_chain_idx[:, :, :, 0:1]).to(dev)

        E *= same_chain.unsqueeze(-1)
        return E


class MultiChainProteinFeatures(ProteinFeatures):
    """ Protein backbone featurization which accounts for differences
    between inter-chain and intra-chain interactions.

    Attributes
    ----------
    embeddings : IndexDiffEncoding
        Module to generate differential positional embeddings for edges
    dropout : nn.Dropout
        Dropout module
    node_embeddings, edge_embeddings : nn.Linear
        Embedding layers for nodes and edges
    norm_nodes, norm_edges : nn.LayerNorm
        Normalization layers for node and edge features
    """
    def __init__(self,
                 edge_features,
                 node_features,
                 num_positional_embeddings=16,
                 num_rbf=16,
                 top_k=30,
                 features_type='full',
                 features_options='',
                 augment_eps=0.,
                 dropout=0.1,
                 chain_handle='',
                 zero_out_pos_embs=False,
                ):
        """ Extract protein features """
        super().__init__(edge_features,
                         node_features,
                         num_positional_embeddings=num_positional_embeddings,
                         num_rbf=num_rbf,
                         top_k=top_k,
                         features_type=features_type,
                         features_options=features_options,
                         augment_eps=augment_eps,
                         dropout=dropout,
                         chain_handle=chain_handle,
                         zero_out_pos_embs=zero_out_pos_embs
                        )

        # so uh this is designed to work on the batched TERMS
        # but if we just treat the whole sequence as one big TERM
        # the math is the same so i'm not gonna code a new module lol
        if num_positional_embeddings > 0:
            self.embeddings = IndexDiffEncoding(num_positional_embeddings)
        else:
            self.embeddings = None

    # pylint: disable=arguments-differ
    def forward(self, X, chain_dict, mask, need_node_embeddings=True):
        """ Featurize coordinates as an attributed graph

        Args
        ----
        X : torch.Tensor
            Backbone coordinates
            Shape: n_batch x seq_len x 4 x 3
        chain_idx : torch.LongTensor
            Indices for residues such that each chain is assigned a unique integer
            and each residue in that chain is assigned that integer
            Shape: n_batches x seq_len
        mask : torch.ByteTensor
            Mask for residues
            Shape: n_batch x seq_len
        inds_reduce : torch.Tensor
            Indexes for converting a per residue per knn representation to a per edge representation
            shape: n_batch x n_edges
        inds_transpose : torch.tensor
            Indexes for positions to transpose in per residue per knn representation
            shape: n_batch x n_duplicated_edges / 2
        inds_duplicate : torch.Tensor
            Indexes for positions to duplicate in per residue per knn representation
            shape: n_batch x n_duplicated_edges / 2
        inds_singles : torch.Tensor
            Indexes for positions to not duplicate (i.e., singleton edges)
            shape: n_batch x n_single_edges

        Returns
        -------
        V : torch.Tensor
            Node embeddings
            Shape: n_batches x seq_len x n_hidden
        E : torch.Tensor
            Edge embeddings in kNN dense form
            Shape: n_batches x seq_len x k x n_hidden
        E_idx : torch.LongTensor
            Edge indices
            Shape: n_batches x seq_len x k x n_hidden
        """
        # Build k-Nearest Neighbors graph
        X_ca = X[:, :, 1, :]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask, chain_ids=chain_dict['ids'])
        # Pairwise features
        if self.features_options.find("residue_local") > -1:
            AD_features, O_features = self._orientations_coarse(X, E_idx, frame="residue")
        else:
            AD_features, O_features = self._orientations_coarse(X_ca, E_idx)
        RBF = self._rbf(D_neighbors)

        if self.features_type == 'coarse':
            # Coarse backbone features
            V = AD_features
            E = torch.cat((RBF, O_features), -1)
        elif self.features_type == 'hbonds':
            # Hydrogen bonds and contacts
            neighbor_HB = self._hbonds(X, E_idx, mask_neighbors)
            neighbor_C = self._contacts(D_neighbors, mask_neighbors)
            # Dropout
            neighbor_C = self.dropout(neighbor_C)
            neighbor_HB = self.dropout(neighbor_HB)
            # Pack
            V = mask.unsqueeze(-1) * torch.ones_like(AD_features)
            neighbor_C = neighbor_C.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            neighbor_HB = neighbor_HB.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            E = torch.cat((neighbor_C, neighbor_HB), -1)
        elif self.features_type == 'full':
            # Full backbone angles
            V = self._dihedrals(X, mask, chain_dict, self.chain_handle)
            E = torch.cat((RBF, O_features), -1)
        elif self.features_type == 'dist':
            # Full backbone angles
            V = self._dihedrals(X, mask, chain_dict, self.chain_handle)
            E = RBF

        # Pairwise embeddings
        # we unsqueeze to generate "1 TERM" per sequence,
        # then squeeze it back to get rid of it
        if self.embeddings is not None:
            E_positional = self.embeddings(E_idx.unsqueeze(1), chain_dict['ids']).squeeze(1)
            if self.zero_out_pos_embs:
                E_positional = E_positional*0
            E = torch.cat((E_positional, E), -1)
        
        if need_node_embeddings:
            V = self.node_embedding(V)
            V = self.norm_nodes(V)
        else:
            V = None
            
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        
        return V, E, E_idx



