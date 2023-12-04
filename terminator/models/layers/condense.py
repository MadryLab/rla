" TERM Information Condensor and submodules """

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .term.matches.attn import TERMMatchTransformerEncoder
from .term.matches.cnn import Conv1DResNet
from .term.graph.s2s import TERMGraphTransformerEncoder
from .utils import aggregate_edges, batchify, cat_term_edge_endpoints

# pylint: disable=no-member

NUM_AA = 21
NUM_FEATURES = len(['sin_phi', 'sin_psi', 'sin_omega', 'cos_phi', 'cos_psi', 'cos_omega', 'env', 'rmsd', 'term_len'])
NUM_TARGET_FEATURES = len(['sin_phi', 'sin_psi', 'sin_omega', 'cos_phi', 'cos_psi', 'cos_omega', 'env'])


class ResidueFeatures(nn.Module):
    """ Module which featurizes TERM match residue information

    Attributes
    ----------
    embedding: nn.Embedding
        Embedding layer for residue identities (represented as int)
    relu: nn.ReLU
        ReLU activation layer
    tanh : nn.Tanh
        tanh activation layer
    lin1, lin2 : nn.Linear
        Embedding layers
    bn : nn.BatchNorm2d
        Batch Normalization for features
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

        self.embedding = nn.Embedding(NUM_AA, hdim - hparams['num_features'])
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.lin1 = nn.Linear(hdim, hdim)
        self.lin2 = nn.Linear(hdim, hdim)
        self.bn = nn.BatchNorm2d(hdim)

    def forward(self, X, features):
        """ Featurize TERM matches and their associated features

        Args
        ----
        X : torch.LongTensor
            Match residue identities
            Shape: n_batches x n_matches x sum_term_len
        features : torch.Tensor
            Features associated with match residues (e.g. torsion angles, RMSD, environment value
            Shape: n_batches x n_matches x sum_term_len x NUM_TERM_FEATURES

        Returns
        -------
        out : torch.Tensor
            Featurized TERM match residues
            Shape: n_batches x n_hidden x sum_term_len x n_alignments
        """
        # X: num batches x num alignments x sum TERM length
        # features: num_batches x num alignments x sum TERM length x num features
        # samples in X are in rows
        embedded = self.embedding(X)

        # hidden dim = embedding hidden dim + num features
        # out: num batches x num alignments x TERM length x hidden dim
        out = torch.cat((embedded, features), dim=3)

        # transpose so that features = number of channels for convolution
        # out: num batches x num channels x TERM length x num alignments
        out = out.transpose(1, 3)

        # normalize over channels (TERM length x num alignments)
        out = self.bn(out)

        # embed features using ffn
        out = out.transpose(1, 3)
        out = self.lin1(out)
        if not self.hparams['res_embed_linear']:
            out = self.relu(out)
            out = self.lin2(out)
            out = self.tanh(out)

        # retranspose so features are channels
        out = out.transpose(1, 3)
        return out


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class ContactIndexEncoding(nn.Module):
    """ Module which sinusoidally embeds contact indices

    Attributes
    ----------
    dropout : nn.Dropout
        Dropout module
    cie_scaling : int
        Multiplicative scaling factor for inputted contact indices
    cie_offset : int
        Additive scaling factor for inputted contact indices
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
        self.dropout = nn.Dropout(p=hparams['cie_dropout'])
        self.hidden_dim = hparams['term_hidden_dim']
        self.cie_scaling = hparams['cie_scaling'] if 'cie_scaling' in hparams else 500  # tested to work
        self.cie_offset = hparams['cie_offset'] if 'cie_offset' in hparams else 0

    def forward(self, contact_idxs, mask=None):
        """ Embed contact indicies sinusoidally

        Args
        ----
        contact_idxs : torch.LongTensor
            Contact indices

        Returns
        -------
        cie : torch.Tensor
            Sinusoidally embedded contact indices
        """
        dev = contact_idxs.device
        hdim = self.hidden_dim
        cie = torch.zeros(list(contact_idxs.shape) + [hdim]).to(dev)
        position = contact_idxs.unsqueeze(-1)
        position = position * self.cie_scaling + self.cie_offset
        div_term = torch.exp(torch.arange(0, hdim, 2).double() * (-math.log(10000.0) / hdim)).to(dev)
        cie[:, :, 0::2] = torch.sin(position * div_term)
        cie[:, :, 1::2] = torch.cos(position * div_term)
        if mask is not None:
            cie = cie * mask.unsqueeze(-1).float()

        return self.dropout(cie)


def covariation_features(matches, term_lens, rmsds, mask):
    """ Compute weighted cross-covariance features from TERM matches

    Args
    ----
    matches : torch.Tensor
        TERM matches, in flat form (TERMs are cat'd side by side)
        Shape: n_batch x sum_term_len x n_hidden
    term_lens : list of (list of int)
        Length of each TERM
    rmsds : torch.Tensor
        RMSD per TERM match
        Shape: n_batch x sum_term_len
    mask : torch.ByteTensor
        Mask for TERM residues
        Shape: n_batch x sum_term_len

    Returns
    -------
    cov_mat : torch.Tensor
        Weighted cross-covariance matrices
        Shape: n_batch x n_terms x max_term_len x max_term_len x n_hidden x n_hidden
    """
    local_dev = matches.device
    batchify_terms = batchify(matches, term_lens)
    term_rmsds = batchify(rmsds, term_lens)
    # try using -rmsd as weight
    term_rmsds = -term_rmsds
    term_rmsds[term_rmsds == 0] = torch.tensor(np.finfo(np.float32).min).to(local_dev)
    weights = F.softmax(term_rmsds, dim=-1)

    weighted_mean = (weights.unsqueeze(-1) * batchify_terms).sum(dim=-2)
    centered = batchify_terms - weighted_mean.unsqueeze(-2)
    weighted_centered = weights.unsqueeze(-1) * centered
    X = weighted_centered.unsqueeze(-3).transpose(-2, -1)
    X_t = weighted_centered.unsqueeze(-4)
    cov_mat = X @ X_t
    mask = mask.unsqueeze(-1).float()
    mask_edges = mask @ mask.transpose(-2, -1)
    mask_edges = mask_edges.unsqueeze(-1).unsqueeze(-1)
    cov_mat *= mask_edges

    return cov_mat


class EdgeFeatures(nn.Module):
    """ Module which computes edge features for TERMs

    Attributes
    ----------
    embedding : nn.Embedding or equivalent, conditionally present
        Layer to embed TERM match residue identities
    lin : nn.Linear, conditionally present
        Input embedding layer
    cnn : Conv2DResNet, conditionally present
        CNN that generates 2D features by convolution over matches
    W : nn.Linear or nn.Sequential(nn.Linear, nn.ReLU, nn.Linear)
        Output layer
    """
    def __init__(self, hparams, in_dim, hidden_dim, feature_mode="shared_learned", compress="project"):
        """
        Args
        ----
        hparams : dict
            Dictionary of model hparams (see :code:`~/scripts/models/train/default_hparams.json` for more info)
        in_dim : int
            Dimensionality of input feature vectors
        hidden_dim : int
            Hidden dimension
        feature_mode : string from :code:`['shared_learned', 'all_raw', 'aa_learned', 'aa_count']`
            Generate initial covariation matrix by computing covariation on
                - :code:`'shared_learned'`: inputted match features without preprocessing
                - :code:`'all_raw'`: raw counts as well as inputted match features
                - :code:`'aa_learned'`: features in learned embedding for residue identity
                - :code:`'aa_count'`: raw residue identity counts
        compress : string from ['project', 'ffn', 'ablate']
            Method to compress covariance matrix to vector. Flatten, then
                - :code:`'project'`: project to proper dimensionality with a linear layer
                - :code:`'ffn'`: use a 2 layer FFN with proper output dimensionality
                - :code:`'ablate'`: return a zero vector of proper dimensionality
        """
        super().__init__()

        self.feature_mode = feature_mode
        self.hparams = hparams

        if feature_mode == "shared_learned":
            pass
        elif feature_mode == "all_raw":
            self.one_hot = torch.eye(NUM_AA)
            self.embedding = lambda x: self.one_hot[x]
            in_dim = NUM_AA + NUM_FEATURES
        elif feature_mode == "all_learned":
            self.one_hot = torch.eye(NUM_AA)
            self.embedding = lambda x: self.one_hot[x]
            in_dim = NUM_AA + NUM_FEATURES
            self.lin = nn.Linear(in_dim, in_dim)
        elif feature_mode == "aa_learned":
            self.embedding = nn.Embedding(NUM_AA, in_dim)
        elif feature_mode == "aa_counts":
            self.one_hot = torch.eye(NUM_AA)
            self.embedding = lambda x: self.one_hot[x]
            in_dim = NUM_AA
        else:
            raise ValueError(f"{feature_mode} is not a valid feature mode for EdgeFeatures")

        if compress == "project":
            self.W = nn.Linear(in_dim**2, hidden_dim, bias=False)
        elif compress == "ffn":
            self.W = nn.Sequential(nn.Linear(in_dim**2, hidden_dim * 4), nn.ReLU(),
                                   nn.Linear(hidden_dim * 4, hidden_dim))
        elif compress == "ablate":
            self.W = torch.zeros_like
        else:
            raise ValueError(f"{compress} is not a valid compression mode for EdgeFeatures")

    def forward(self, matches, term_lens, rmsds, mask, features=None):
        """ Generate embeddings for weighted covariation features between TERM residues

        Args
        ----
        matches : torch.Tensor
            Matches, either as ints representing TERM match residue identities,
            or featurized matches
            Shape: n_batches x sum_term_len (x n_in if already featurized)
        term_lens : list of (list of int)
            Length of TERMs per protein
        rmsds : torch.Tensor
            RMSD associated with each match
            Shape: n_batches x sum_term_len
        mask : torch.ByteTensor
            Mask for TERM residues
            Shape: n_batches x sum_term_len

        Returns
        -------
        cov_features : torch.Tensor
            Embeddings for covariance matrices between TERM residues
            Shape: n_batches x n_terms x max_term_len x max_term_len x n_hidden
        """
        feature_mode = self.feature_mode
        if feature_mode in ('aa_counts', 'aa_learned', "all_raw", "all_learned"):
            local_dev = matches.device
            matches = self.embedding(matches).to(local_dev)
            if feature_mode == "all_raw":
                assert features is not None, "features should not be None!"
                matches = torch.cat([matches, features], -1)
            elif feature_mode == "all_learned":
                assert features is not None, "features should not be None!"
                matches = torch.cat([matches, features], -1)
                matches = self.lin(matches)

        if feature_mode != "preprocessed":
            cov_mat = covariation_features(matches, term_lens, rmsds, mask)
        else:
            cov_mat = matches

        n_batch, n_term, n_aa = cov_mat.shape[:3]
        cov_features = cov_mat.view([n_batch, n_term, n_aa, n_aa, -1])
        return self.W(cov_features)


class CondenseTERM(nn.Module):
    """ TERM Information Condensor

    Condense TERM matches and aggregate them together to form a full structure embedding

    Attributes
    ----------
    embedding : ResidueFeatures
        Feature embedding module for TERM match residues
    edge_features : EdgeFeatures
        Feature embedding module for TERM match residue interactions
    matches : Conv1DResNet, TERMMatchTransformerEncoder, or None
        Matches Condensor (reduce the matches into a singular embedding per TERM residue)
    W_ppoe : nn.Linear
        Linear layer for target structural features (e.g. featurized torsion angles, RMSD, environment values)
    term_mpnn : TERMGraphTransformerEncoder
        TERM MPNN (refine TERM graph embeddings)
    cie : ContactIndexEncoding, present when :code:`hparams['contact_idx']=True`
        Sinusoidal encoder for contact indices
    W_v, W_e : nn.Linear, present when :code:`hparams['term_mpnn_linear']=True`
        Modules to linearize TERM MPNN
    """
    def __init__(self, hparams, device='cuda:0'):
        """
        Args
        ----
        hparams : dict
            Dictionary of model hparams (see :code:`~/scripts/models/train/default_hparams.json` for more info)
        device : str, default='cuda:0'
            What device to place the module on
        """
        super().__init__()
        self.hparams = hparams
        h_dim = hparams['term_hidden_dim']
        self.num_sing_stats = hparams['num_sing_stats']
        self.num_pair_stats = hparams['num_pair_stats']
        self.embedding = ResidueFeatures(hparams=self.hparams)

        # configure edge embeddings
        if hparams['cov_features']:
            if self.num_pair_stats:
                in_dim = self.num_pair_stats
            else:
                in_dim = h_dim
            self.edge_features = EdgeFeatures(hparams,
                                              in_dim=in_dim,
                                              hidden_dim=h_dim,
                                              feature_mode=hparams['cov_features'],
                                              compress=hparams['cov_compress'])
        else:
            raise ValueError("'cov_features' must be specified in TERMinator")

        # Matches Condensor
        if hparams['matches'] == 'resnet':
            self.matches = Conv1DResNet(hparams=self.hparams)
        elif hparams['matches'] == 'transformer':
            self.matches = TERMMatchTransformerEncoder(hparams=hparams)
            # project target ppoe to hidden dim
            self.W_ppoe = nn.Linear(NUM_TARGET_FEATURES, h_dim)
        elif hparams['matches'] == 'ablate':
            pass  # we don't need this module if it's ablated
        else:
            raise ValueError(f"arg for matches condenser {hparams['matches']} doesn't look right")

        # construct TERM MPNN
        # if contact idxs are enabled, include the CIE
        if hparams['contact_idx']:
            self.cie = ContactIndexEncoding(hparams=self.hparams)
        # to linearize TERM transformer
        if hparams['term_mpnn_linear']:
            self.W_v = nn.Linear(2 * h_dim, h_dim)
            self.W_e = nn.Linear(3 * h_dim, h_dim)
        else:
            self.term_mpnn = TERMGraphTransformerEncoder(hparams=self.hparams)

        if torch.cuda.is_available():
            self.dev = device
        else:
            print('No CUDA device detected. Defaulting to cpu')
            self.dev = 'cpu'

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # TODO: check shapes in docstring
    def _matches(self, embeddings, ppoe, focuses, src_key_mask):
        """ Extract singleton statistics from matches using MatchesCondensor

        Args
        ----
        embeddings : torch.Tensor
            Embedded match features
            Shape: TODO

        ppoe : torch.Tensor
            Target structure :math:`\\phi, \\psi, \\omega`, and environment value
            Shape: n_batch x seq_len x 4

        focuses : torch.LongTensor
            Integer indices corresponding to `embeddings` which specifies
            what residue in the target structure that set of matches corresponds to
            Shape: TODO

        Returns
        -------
        condensed_matches : torch.Tensor
            The condensed matches, such that each term residue has one vector associated with it
        """
        # use Convolutional ResNet or Transformer
        # for further embedding and to reduce dimensionality
        if self.hparams['matches_linear']:
            condensed_matches = embeddings.mean(dim=-1).transpose(1, 2)
        elif self.hparams['matches'] == 'transformer':
            # project target ppoe
            ppoe = self.W_ppoe(ppoe)
            # gather to generate target ppoe per term residue
            focuses_gather = focuses.unsqueeze(-1).expand(-1, -1, self.hparams['term_hidden_dim'])
            target = torch.gather(ppoe, 1, focuses_gather)

            # output dimensionality of embeddings is different for transformer
            condensed_matches = self.matches(embeddings.transpose(1, 3).transpose(1, 2), target, ~src_key_mask)
        elif self.hparams['matches'] == 'resnet':
            condensed_matches = self.matches(embeddings)
        elif self.hparams['matches'] == 'ablate':
            condensed_matches = torch.zeros_like(embeddings.mean(dim=-1).transpose(1, 2))

        return condensed_matches

    def _edges(self, embeddings, features, X, term_lens, batched_focuses, batchify_src_key_mask):
        """ Compute edge embeddings for TERMs

        TODO: check shapes

        Args
        ----
        embeddings : torch.Tensor, conditionally used
            Featurized matches
            Shape: TODO
        features : torch.Tensor
            TERM match residue features (e.g. sinusoidally embedded torsion angles, rmsd, environment value)
            RMSD should be at index 7.
            Shape: TODO
        X : torch.LongTensor, conditionally used
            Raw TERM match residue identities
            Shape: n_batches x n_matches x sum_term_len
        term_lens : list of (list of int)
            Length of TERMs per protein
        batched_focuses : torch.LongTensor
            Sequence position indices for TERM residues, batched by TERM
            Shape: TODO
        batchify_src_key_mask : torch.ByteTensor
            Mask for TERM residues, batched by TERM
            Shape: TODO

        Returns
        -------
        edge_features : torch.Tensor
            TERM edge features
            Shape: n_batches x n_terms x max_term_len x max_term_len x n_hidden
        batch_rel_E_idx : torch.LongTensor
            Edge indices within a TERM
            Shape: n_batches x n_terms x max_term_len x max_term_len x n_hidden
        batch_abs_E_idx : torch.LongTensor
            Edge indices relative to the target structure
            Shape: n_batches x n_terms x max_term_len x max_term_len x n_hidden
        """
        local_dev = embeddings.device
        cv = self.hparams['cov_features']
        if cv in ['shared_learned', 'cnn']:
            # generate covariation features
            embeddings = embeddings.transpose(1, 3).transpose(1, 2)
        elif cv in ['aa_learned', 'aa_counts', "all_raw", "all_learned"]:
            embeddings = X.transpose(-2, -1)

        rmsds = features[..., 7].transpose(-2, -1)
        edge_features = self.edge_features(embeddings,
                                           term_lens,
                                           rmsds,
                                           batchify_src_key_mask,
                                           features=features.transpose(1, 2))

        # edge features don't have the self edge as the first element in the row
        # so we need to rearrange the edges so they are
        # we'll use a shifted E_idx to do this
        num_batch = edge_features.shape[0]
        max_num_terms = max([len(l) for l in term_lens])
        max_term_len = edge_features.shape[2]
        shift_E_idx_slice = torch.arange(max_term_len).unsqueeze(0).repeat([max_term_len, 1])
        for i in range(max_term_len):
            shift_E_idx_slice[i][:i+1] = i - torch.arange(i+1)
        batch_rel_E_idx = shift_E_idx_slice.view([1, 1, max_term_len,
                                                  max_term_len]).expand([num_batch, max_num_terms, -1,
                                                                         -1]).contiguous().to(local_dev)
        # use gather to rearrange the edge features
        edge_features = torch.gather(
            edge_features, -2,
            batch_rel_E_idx.unsqueeze(-1).expand(list(batch_rel_E_idx.shape) + [self.hparams['term_hidden_dim']]))

        # we need an absolute version of the rel_E_idx so we can aggregate edges
        batch_abs_E_idx = torch.gather(
            batched_focuses.unsqueeze(-2).expand(-1, -1, max_term_len, -1), -1, batch_rel_E_idx)

        return edge_features, batch_rel_E_idx, batch_abs_E_idx

    def _term_mpnn(self,
                   batchify_terms,
                   edge_features,
                   batch_rel_E_idx,
                   src_key_mask,
                   term_lens=None,
                   contact_idx=None):
        """ Run TERM MPNN to refine graph embeddings

        Args
        ----
        batchify_terms : torch.Tensor
            TERM residue node features
            Shape: n_batches x n_terms x max_term_len x n_hidden
        edge_features : torch.Tensor
            TERM residue interaction features
            Shape: n_batches x n_terms x max_term_len x max_term_len x n_hidden
        batch_rel_E_idx : torch.LongTensor
            Edge indices local to each TERM graph
            Shape: n_batches x n_terms x max_term_len x max_term_len
        src_key_mask : torch.ByteTensor
            Mask for TERM residues
            Shape: n_batches x sum_term_len
        term_lens : list of (list of int)
            Length of TERMs per protein
        contact_idx : torch.Tensor
            Contact indices per TERM residue
            Shape: n_batches x sum_term_len

        Returns
        -------
        node_embeddings : torch.Tensor
            Updated TERM residues embeddings
            Shape: n_batches x n_terms x max_term_len x n_hidden
        edge_embeddings : torch.Tensor
            Updated TERM residue interaction embeddings
            Shape: n_batches x n_terms x max_term_len x max_term_len x n_hidden
        """
        batchify_src_key_mask = batchify(~src_key_mask, term_lens)
        if self.hparams['contact_idx']:
            assert contact_idx is not None
            assert term_lens is not None
            contact_idx = self.cie(contact_idx, ~src_key_mask)
            contact_idx = batchify(contact_idx, term_lens)
            if not self.hparams['term_mpnn_linear']:
                # big transform
                node_embeddings, edge_embeddings = self.term_mpnn(batchify_terms,
                                                                  edge_features,
                                                                  batch_rel_E_idx,
                                                                  mask=batchify_src_key_mask.float(),
                                                                  contact_idx=contact_idx)
            else:
                node_embeddings = self.W_v(torch.cat([batchify_terms, contact_idx], dim=-1))
                node_embeddings *= batchify_src_key_mask.unsqueeze(-1)
                edge_embeddings = self.W_e(cat_term_edge_endpoints(edge_features, contact_idx, batch_rel_E_idx))
                mask = batchify_src_key_mask.unsqueeze(-1).float()
                edge_mask = mask @ mask.transpose(-1, -2)
                edge_embeddings *= edge_mask.unsqueeze(-1)

        else:
            node_embeddings, edge_embeddings = self.term_mpnn(batchify_terms,
                                                              edge_features,
                                                              batch_rel_E_idx,
                                                              mask=batchify_src_key_mask.float())
        return node_embeddings, edge_embeddings

    def _agg_nodes(self, node_embeddings, batched_focuses, seq_lens, n_batches, max_seq_len):
        """ Fuse together TERM match residues so that every residue has one embedding.

        Args
        ----
        node_embeddings : torch.Tensor
            TERM residue embeddings
            Shape: n_batches x n_terms x max_term_len x n_hidden
        batched_focuses : torch.LongTensor
            Indices for which full-structure residue corresponds to the TERM match residue
            Shape: n_batches x n_terms x max_term_len
        seq_lens : list of int
            Protein lengths in the batch
        n_batches : int
            Number of batches
        max_seq_len : int
            Maximum length of proteins in the batch

        Returns
        -------
        aggregate : torch.Tensor
            Residue embeddings derived from TERM data
            Shape: n_batches x max_seq_len x n_hidden
        """
        local_dev = node_embeddings.device
        # create a space to aggregate term data
        aggregate = torch.zeros((n_batches, max_seq_len, self.hparams['term_hidden_dim'])).to(local_dev)
        count = torch.zeros((n_batches, max_seq_len, 1)).to(local_dev).long()

        # this make sure each batch stays in the same layer during aggregation
        layer = torch.arange(n_batches).unsqueeze(-1).unsqueeze(-1).expand(batched_focuses.shape).long().to(local_dev)

        # aggregate node embeddings and associated counts
        aggregate = aggregate.index_put((layer, batched_focuses), node_embeddings, accumulate=True)
        count_idx = torch.ones_like(batched_focuses).unsqueeze(-1).to(local_dev)
        count = count.index_put((layer, batched_focuses), count_idx, accumulate=True)

        # set all the padding zeros in count to 1 so we don't get nan's from divide by zero
        batch_zeros = []

        for batch, sel in enumerate(seq_lens):
            count[batch, sel:] = 1
            if (count[batch] == 0).any():
                batch_zeros.append(batch)
        if len(batch_zeros) > 0:
            raise RuntimeError(
                f"entries {batch_zeros} should have nonzero count but count[batches] is {count[batch_zeros]}")

        # average the aggregate
        aggregate = aggregate / count.float()
        return aggregate

    def forward(self, data, max_seq_len):
        """ Convert input TERM data into a full structure representation

        Args
        ----
        data : dict
            Input data dictionary. See :code:`~/terminator/data/data.py` for more information.
        max_seq_len : int
            Length of the largest protein in the input data

        Returns
        -------
        agg_nodes : torch.Tensor
            Structure node embedding
            Shape: n_batch x max_seq_len x n_hidden
        agg_edges : torch.Tensor
            Structure edge embeddings
            Shape: n_batch x max_seq_len x max_seq_len x n_hidden
        """
        # grab necessary data
        X = data['msas']
        features = data['features']
        seq_lens = data['seq_lens']
        focuses = data['focuses']
        term_lens = data['term_lens']
        src_key_mask = data['src_key_mask']
        ppoe = data['ppoe']
        contact_idx = data['contact_idxs']

        # some batch management number manipulation
        n_batches = X.shape[0]
        seq_lens = seq_lens.tolist()
        term_lens = term_lens.tolist()
        for i, _ in enumerate(term_lens):
            for j, _ in enumerate(term_lens[i]):
                if term_lens[i][j] == -1:
                    term_lens[i] = term_lens[i][:j]
                    break
        local_dev = X.device

        # zero out all positions used as padding so they don't contribute to aggregation
        negate_padding_mask = (~src_key_mask).unsqueeze(-1).expand(-1, -1, self.hparams['term_hidden_dim'])
        # embed MSAs and concat other features on
        embeddings = self.embedding(X, features)

        # apply Matches Condensor
        condensed_matches = self._matches(embeddings, ppoe, focuses, src_key_mask)

        # zero out biases introduced into padding
        condensed_matches *= negate_padding_mask
        # reshape batched flat terms into batches of terms
        batchify_terms = batchify(condensed_matches, term_lens)
        # also reshape the mask
        batchify_src_key_mask = batchify(~src_key_mask, term_lens)
        # we also need to batch focuses to we can aggregate data
        batched_focuses = batchify(focuses, term_lens).to(local_dev)

        # generate edge features
        edge_features, batch_rel_E_idx, batch_abs_E_idx = self._edges(embeddings, features, X, term_lens,
                                                                      batched_focuses, batchify_src_key_mask)
        # run TERM MPNN
        node_embeddings, edge_embeddings = self._term_mpnn(batchify_terms,
                                                           edge_features,
                                                           batch_rel_E_idx,
                                                           src_key_mask,
                                                           term_lens=term_lens,
                                                           contact_idx=contact_idx)
        # aggregate nodes and edges using batch_abs_E_idx
        agg_nodes = self._agg_nodes(node_embeddings, batched_focuses, seq_lens, n_batches, max_seq_len)
        agg_edges = aggregate_edges(edge_embeddings, batch_abs_E_idx, max_seq_len)

        return agg_nodes, agg_edges
