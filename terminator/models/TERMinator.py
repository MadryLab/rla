"""TERMinator models"""
import torch
import torch_geometric.data
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .layers.condense import CondenseTERM
from .layers.energies.gvp import GVPPairEnergies
from .layers.energies.s2s import (AblatedPairEnergies, PairEnergies, TransformerPairEnergies)
from .layers.utils import gather_edges, pad_sequence_12

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

        if self.hparams["use_terms"]:
            self.hparams['energies_input_dim'] = self.hparams['term_hidden_dim']
            self.bot = CondenseTERM(hparams, device=self.dev)
        else:
            self.hparams['energies_input_dim'] = 0

        if hparams['struct2seq_linear']:
            self.top = AblatedPairEnergies(hparams).to(self.dev)
        elif hparams['energies_gvp']:
            self.top = GVPPairEnergies(hparams).to(self.dev)
        else:
            if hparams.get('energies_style', 'mpnn') == 'graphformer':
                self.top = TransformerPairEnergies(hparams).to(self.dev)
            else:
                self.top = PairEnergies(hparams).to(self.dev)

        if self.hparams['use_terms']:
            print(f'TERM information condenser hidden dimensionality is {self.bot.hparams["term_hidden_dim"]}')
        print(f'GNN Potts Model Encoder hidden dimensionality is {self.top.hparams["energies_hidden_dim"]}')

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _to_gvp_input(self, node_embeddings, edge_embeddings, data):
        """ Convert Ingraham-style inputs to Jing-style inputs for use in GVP models

        Args
        ----
        node_embeddings : torch.Tensor or None
            Node embeddings at the structure level, outputted by the TERM Info Condensor.
            :code:`None` if running in TERMless mode
            Shape: n_batch x max_seq_len x tic_n_hidden

        edge_embeddings : torch.Tensor or None
            Edge embedings at the structure level, outputted by the TERM Info Condensor.
            :code:`None` if running in TERMless mode
            Shape: n_batch x max_seq_len x max_seq_len x tic_n_hidden

        data : dict of torch.Tensor
            Overall input data dictionary. See :code:`forward` for more info.

        Returns
        -------
        h_V : torch.Tensor
            Node embeddings in Jing format
        edge_idex : torch.LongTensor
            Edge index matrix in Jing format (sparse form)
        h_E : torch.Tensor
            Edge embeddings in Jing format
        E_idx : torch.LongTensor
            Edge index matrix in Ingraham format (kNN form)
        """
        gvp_data_list = [data['gvp_data'][i] for i in data['scatter_idx'].tolist()]
        gvp_batch = torch_geometric.data.Batch.from_data_list(gvp_data_list)
        seq_lens = data['seq_lens']
        # extract node_embeddings and flatten
        if node_embeddings is not None:
            gvp_batch = gvp_batch.to(node_embeddings.device)
            node_embeddings = torch.cat([h_V[:seq_lens[i]] for i, h_V in enumerate(torch.unbind(node_embeddings))],
                                        dim=0)

            h_V = (torch.cat([gvp_batch.node_s, node_embeddings], dim=-1), gvp_batch.node_v)
        else:
            h_V = (gvp_batch.node_s, gvp_batch.node_v)


        # compute global E_idx from edge_index
        total_len = seq_lens.sum()
        batched_E_idx = gvp_batch.edge_index[0].view(total_len, self.hparams['k_neighbors'])
        split_E_idxs = torch.split(batched_E_idx, list(seq_lens))
        offset = [sum(seq_lens[:i]) for i in range(len(seq_lens))]
        split_E_idxs = [e.to(seq_lens.device) - offset for e, offset in zip(split_E_idxs, offset)]
        E_idx = pad_sequence(split_E_idxs, batch_first=True)
        if edge_embeddings is not None:
            # gather relevant edges
            E_embed_neighbors = gather_edges(edge_embeddings, E_idx)

            # flatten edge_embeddings
            edge_embeddings_source = torch.cat(
                [h_E[:seq_lens[i]] for i, h_E in enumerate(torch.unbind(E_embed_neighbors))], dim=0)
            edge_embeddings_flat = edge_embeddings_source.view(
                [gvp_batch.edge_index.shape[1], self.hparams['term_hidden_dim']])

            h_E = (torch.cat([gvp_batch.edge_s, edge_embeddings_flat], dim=-1), gvp_batch.edge_v)
        else:
            h_E = (gvp_batch.edge_s, gvp_batch.edge_v)

        return h_V, gvp_batch.edge_index, h_E, E_idx

    def _from_gvp_outputs(self, h_E, E_idx, seq_lens, max_seq_len):
        """ Convert outputs of GVP models to Ingraham style outputs

        Args
        ----
        h_E : torch.Tensor
            Outputted Potts Model in Jing format
        E_idx : torch.Tensor
            Edge index matrix in Ingraham format (kNN sparse)
        seq_lens : np.ndarray (int)
            Sequence lens of proteins in batch
        max_seq_len : int
            Max sequence length of proteins in batch

        Returns
        -------
        etab : torch.Tensor
            Potts Model in Ingraham Format
        E_idx : torch.LongTensor
            Edge index matrix in Ingraham format (kNN sparse)
        """
        # convert gvp outputs to TERMinator format
        h_E = h_E.view([
            h_E.shape[0] // self.hparams['k_neighbors'], self.hparams['k_neighbors'],
            self.hparams['energies_output_dim']
        ])
        split_h_E = torch.split(h_E, seq_lens.tolist())
        etab = pad_sequence_12(split_h_E)

        #print(etab.shape, E_idx.shape)
        # pad the difference if using DataParallel
        padding_diff = max_seq_len - etab.shape[1]
        if padding_diff > 0:
            padding = torch.zeros(etab.shape[0], padding_diff, etab.shape[2], etab.shape[3], device=etab.device)
            etab = torch.cat([etab, padding], dim=1)
            padding = torch.zeros(etab.shape[0], padding_diff, etab.shape[2], device=etab.device).long()
            E_idx = torch.cat([E_idx, padding], dim=1)

        return etab, E_idx

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
            term_lens : int np.ndarray
                2D Array of batched TERM lengths.
                Shape: n_batch x n_terms
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
            gvp_data : list of torch_geometric.data.Data
                Vector and scalar featurizations of the backbone, as required by GVP
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
        if self.hparams['use_terms']:
            node_embeddings, edge_embeddings = self.bot(data, max_seq_len)
        elif init_node_embeddings is not None:
            node_embeddings = init_node_embeddings
            edge_embeddings = None
        else:
            node_embeddings, edge_embeddings = None, None
        final_node_embeddings = None
        if self.hparams['energies_gvp']:
            h_V, edge_index, h_E, E_idx = self._to_gvp_input(node_embeddings, edge_embeddings, data)
            h_E, edge_index = self.top(h_V, edge_index, h_E)

            etab, E_idx = self._from_gvp_outputs(h_E, E_idx, data['seq_lens'], max_seq_len)
        else:
            etab, final_node_embeddings, E_idx = self.top(node_embeddings, edge_embeddings, data['X'], data['x_mask'], data['chain_dict'])

        if self.hparams['k_cutoff']:
            k = E_idx.shape[-1]
            k_cutoff = self.hparams['k_cutoff']
            assert k > k_cutoff > 0, f"k_cutoff={k_cutoff} must be greater than k"
            etab = etab[..., :k_cutoff, :]
            E_idx = E_idx[..., :k_cutoff]

        return etab, E_idx, final_node_embeddings
