"""TERMinator models"""
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .layers.energies.s2s import (AblatedPairEnergies, PairEnergies, TransformerPairEnergies)

# pylint: disable=no-member, not-callable


class TERMinator(nn.Module):
    """TERMinator model for multichain proteins

    Attributes
    ----------
    dev: str
        Device representing where the model is held
    hparams: dict
        Dictionary of parameter settings (see :code:`terminator/utils/model/default_hparams.py`)
    bot: CondenseTERM
        TERM information condenser network
    top: PairEnergies (or appropriate variant thereof)
        GNN Potts Model Encoder network
    """
    def __init__(self, hparams, device='cuda:0'):
        """
        Initializes TERMinator according to given parameters.

        Args
        ----
        hparams : dict
            Dictionary of parameter settings (see :code:`terminator/utils/model/default_hparams.py`)
        device : str
            Device to place model on
        """
        super().__init__()
        self.dev = device
        self.hparams = hparams
        self.hparams['energies_input_dim'] = 0

        if hparams['struct2seq_linear']:
            self.top = AblatedPairEnergies(hparams).to(self.dev)
        else:
            if hparams.get('energies_style', 'mpnn') == 'graphformer':
                self.top = TransformerPairEnergies(hparams).to(self.dev)
            else:
                self.top = PairEnergies(hparams).to(self.dev)
        print(f'GNN Potts Model Encoder hidden dimensionality is {self.top.hparams["energies_hidden_dim"]}')

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data, max_seq_len=None, init_node_embeddings=None):
        """Compute the Potts model parameters for the structure

        Runs the full TERMinator network for prediction.

        Args
        ----
        data : dict
            Contains the following keys:

            msas : torch.LongTensor
                Integer encoding of sequence matches.
                Shape: n_batch x n_term_res x n_matches
            features : torch.FloatTensor
                Featurization of match structural data.
                Shape: n_batch x n_term_res x n_matches x n_features(=9 by default)
            seq_lens : int np.ndarray
                1D Array of batched sequence lengths.
                Shape: n_batch
            focuses : torch.LongTensor
                Indices for TERM residues matches.
                Shape: n_batch x n_term_res
            src_key_mask : torch.ByteTensor
                Mask for TERM residue positions padding.
                Shape: n_batch x n_term_res
            X : torch.FloatTensor
                Raw coordinates of protein backbones.
                Shape: n_batch x n_res x 4 x 3
            x_mask : torch.ByteTensor
                Mask for X.
                Shape: n_batch x n_res
            sequence : torch.LongTensor
                Integer encoding of ground truth native sequences.
                Shape: n_batch x n_res
            max_seq_len : int
                Max length of protein in the batch.
            ppoe : torch.FloatTensor
                Featurization of target protein structural data.
                Shape: n_batch x n_res x n_features(=9 by default)
            chain_idx : torch.LongTensor
                Integers indices that designate ever residue to a chain.
                Shape: n_batch x n_res
            contact_idx : torch.LongTensor
                Integers representing contact indices across all TERM residues.
                Shape: n_batch x n_term_res
            inds_convert : tuple of torch.Tensor
                    Indexes needed to convert from expanded (directed) to reduced (undirected) dimensionalities
            mask_reduced : torch.ByteTensor
                Mask in reduced dimensionality
        init_node_embeddings : torch.Tensor
            Optional initial node embeddings
            Shape: n_batch x n_res x n_hidden

        Returns
        -------
        etab : torch.FloatTensor
            Dense kNN representation of the energy table, with :code:`E_idx`
            denotating which energies correspond to which edge.
            Shape: n_batch x n_res x k(=30 by default) x :code:`hparams['energies_output_dim']` (=400 by default)
        final_node_embeddings : torch.FloatTensor or None
            Optionally return node representations
            Shape: n_batch x n_res x n_hidden
        E_idx : torch.LongTensor
            Indices representing edges in the kNN graph.
            Given node `res_idx`, the set of edges centered around that node are
            given by :code:`E_idx[b_idx][res_idx]`, with the `i`-th closest node given by
            :code:`E_idx[b_idx][res_idx][i]`.
            Shape: n_batch x n_res x k(=30 by default)
        """

        if init_node_embeddings is not None:
            node_embeddings = init_node_embeddings
            edge_embeddings = None
        else:
            node_embeddings, edge_embeddings = None, None
        final_node_embeddings = None

        etab, final_node_embeddings, E_idx = self.top(node_embeddings, edge_embeddings, data['X'], data['x_mask'], data['chain_dict'])

        if self.hparams['k_cutoff']:
            k = E_idx.shape[-1]
            k_cutoff = self.hparams['k_cutoff']
            assert k > k_cutoff > 0, f"k_cutoff={k_cutoff} must be greater than k"
            etab = etab[..., :k_cutoff, :]
            E_idx = E_idx[..., :k_cutoff]

        return etab, E_idx, final_node_embeddings
