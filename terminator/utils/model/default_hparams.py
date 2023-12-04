""" Default hyperparameter set for TERMinator

Parameters
==========
    matches : str, default='transformer'
        How to processes singleton statistics.

        Options
            'resnet'
                process using a convolutional ResNet
            'transformer'
                process using MatchAttention
            'ablate'
                perform no processing

    term_hidden_dim : int, default=32
        Hidden dimensionality for TERM Information Condenser
        (e.g. net1, self.bot, or CondenseMSA variant)

    enrgies_hidden_dim : int, default=32
        Hidden dimensionality for GNN Potts Model Encoder
        (e.g. net2, self.top, or PairEnergies variant)

    gradient_checkpointing : bool, default=True
        Enable gradient checkpointing at most
        memory-intensive steps (currently, MatchAttention)

    cov_features : str, default='all_raw'
        What features to include for covariance matrix computation. See
        :code:`terminator.models.layers.condense.EdgeFeatures` for more
        information.

        Options
            'shared_learned':
                use those produced by the :code:`ResidueFeatures` module.
            'all_raw':
                concatenate a one-hot encoding of residue identity and the fed-in additional features.
            'all_learned':
                'all_raw', but fed through a dense layer.
            'aa_learned':
                use an embedding matrix for residue identity
            'aa_counts':
                use a one-hot encoding of residue identity
            'cnn':
                use a 2D convolutional neural net on fed-in matches
                (WARNING: not tested due to extreme memory consumption)

    cov_compress : str, default='ffn'
        The method the covariance matrix is compressed into a vector.

        Options
            'ffn'
                Use a 2-layer dense network
            'project'
                Use a linear transformation
            'ablate'
                Use a 0 vector

    num_pair_stats : int, default=28
        [DEPRECIATED] Number of precomputed pairwise match statistics fed into TERMinator

    num_sing_stats : int, default=0
        [DEPRECIATED] Number of precomputed singleton match statistics fed into TERMinator

    resnet_blocks : int, default=4
        Number of ResNet blocks to use if :code:`matches='resnet'`

    term_layers : int, default=4
        Number of TERM MPNN layers to use.

    term_heads : int, default=4
        Number of heads to use in TERMAttention if :code:`term_use_mpnn=False`

    conv_filter : int, default=3
        Length of convolutional filter if code:`matches='resnet'`

    matches_layers : int, default=4
        Number of Transformer layers to use in MatchesCondensor if :code:`matches='transformer'`

    matches_num_heads : int, default=4
        Number of heads to use in MatchAttention if :code:`matches='transformer'`

    k_neighbors : int, default=30
        What `k` is for kNN computation

    k_cutoff : int, default=None
        When outputting a kNN potts model, take the top :code:`k_cutoff` edges and output the truncated etab

    contact_idx : bool, default=True
        Whether or not to include contact indices in computation

    cie_dropout : float, default=0.1
        Dropout rate for sinusoidal encoding of contact index

    cie_scaling : int, default=500
        Multiplicative factor by which to scale contact indices

    cie_offset : int, default=0
        Additive factor by which to offset contact indices

    transformer_dropout : float, default=0.1
        Dropout rate for Transformers used in the TERM Information Condensor

    term_use_mpnn : bool, default=True
        If set to :code:`True`, use a feedforward network to compute TERM graph messages.
        Otherwise, update TERM graph representations using an Attention-based mechanism.

    energies_protein_features : str, default='full'
        Feature set for coordinates fed into the GNN Potts Model Encoder

    energies_augment_eps : float, default=0
        Scaling factor for Gaussian noise added to coordinates before featurization

    energies_encoder_layers : int, default=6
        Number of {node_update, edge_update} layers to include in the GNN Potts Model Encoder

    energies_dropout : float, default=0.1
        Dropout rate in the GNN Potts Model Encoder

    energies_type : one of MPNN, gnn_transformer, graphformer. default MPNN

    energies_output_dim : int, default=400
        Output dimension of GNN Potts Model Encoder

    energies_geometric : bool, default=False
        Use Torch Geometric version of GNN Potts Model Encoder instead

    energies_gvp : bool, default=False
        Use GVP version of GNN Potts Model Encoder instead

    energies_full_graph : bool, default=True
        [DEPRECIATED] Update both node and edge representations in the GNN Potts Model Encoder.
        GNN Potts Model Encoder always updates node and edge representations now,
        making this option do nothing.

    res_embed_linear : bool, default=False
        Replace the singleton matches residue embedding layer with a linear layer.

    matches_linear : bool, default=False
        Remove the Matches Condensor

    term_mpnn_linear : bool, default=False
        Remove the TERM MPNN

    struct2seq_linear : bool, default=False
        Linearize the GNN Potts Model Encoder

    use_terms : bool, default=True
        Whether or not to use the TERM Information Condensor / net1

    term_matches_cutoff : int or None, default=None
        Use the top :code:`term_matches_cutoff` TERM matches for featurization.
        If :code:`None`, apply no cutoff.

    test_term_matches_cutoff : int, optional
        Apply a different :code:`term_matches_cutoff` for validation/evaluation

    use_coords : bool, default=True
        Whether or not to use coordinate-based features in the GNN Potts Model Encoder

    train_batch_size : int or None, default=16
        Batch size for training

    shuffle : bool, default=True
        Whether to do a complete shuffle of the data

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

    regularization : float, default=0
        Amount of L2 regularization to apply to the internal Adam optimizer

    max_term_res : int or None, default=55000
        When :code:`train_batch_size=None, max_term_res>0, max_seq_tokens=None`,
        batch by fitting as many datapoints as possible with the total number of
        TERM residues included below `max_term_res`.
        Calibrated using :code:`nn.DataParallel` on two V100 GPUs.

    max_seq_tokens : int or None, default=None
        When :code:`train_batch_size=None, max_term_res=None, max_seq_tokens>0`,
        batch by fitting as many datapoints as possible with the total number of
        sequence residues included below `max_seq_tokens`.

    term_dropout : str or None, default=None
        Let `t` be the number of TERM matches in the given datapoint.
        Select a random int `n` from 1 to `t`, and take a random subset `n`
        of the given TERM matches to keep. If :code:`term_dropout='keep_first'`,
        keep the first match and choose `n-1` from the rest.
        If :code:`term_dropout='all'`, choose `n` matches from all matches.

    num_features : int, default=9
        The number of non-sequence TERM-based features included per TERM residue.

    loss_config : dict of str (loss component) -> float (scaling factor)
        Dictionary that describes how to construct a loss function. An example dictionary follows:
        .. code-block :
            {
                'nlcpl': 1,
                'etab_norm_penalty': 0.01
            }

    finetune : bool
        Whether or not to train the model in finetuning mode (i.e. freezing all weights but the output layer)
"""
DEFAULT_MODEL_HPARAMS = {
    'num_positional_embeddings': 16,
    'zero_out_pos_embs': False,
    'model': 'multichain',
    'matches': 'transformer',
    'term_hidden_dim': 32,
    'energies_hidden_dim': 32,
    'gradient_checkpointing': True,
    'cov_features': 'all_raw',
    'cov_compress': 'ffn',  #
    'num_pair_stats': 28,  #
    'num_sing_stats': 0,  #
    'resnet_blocks': 4,  #
    'term_layers': 4,  #
    'term_heads': 4,  #
    'conv_filter': 3,  #
    'matches_layers': 4,  #
    'matches_num_heads': 4,  #
    'k_neighbors': 30,  #
    'k_cutoff': None,  #
    'contact_idx': True,  #
    'cie_dropout': 0.1,  #
    'cie_scaling': 500,  #
    'cie_offset': 0,  #
    'transformer_dropout': 0.1,  #
    'term_use_mpnn': True,  #
    'energies_protein_features': 'full',  #
    'energies_augment_eps': 0,  #
    'energies_encoder_layers': 6,  #
    'energies_dropout': 0.1,  #
    'energies_type': 'MPNN',
    'energies_output_dim': 20 * 20,  #
    'energies_gvp': False,  #
    'energies_geometric': False,  #
    'energies_full_graph': True,  #
    'res_embed_linear': False,  #
    'matches_linear': False,  #
    'term_mpnn_linear': False,  #
    'struct2seq_linear': False,
    'use_terms': True,  #
    'use_coords': True,
    'graphformer_config': {"num_layers": 5, "num_heads": 4, "embed_per_head": 64, "mlp_multiplier": 4},
    'num_features':
    len(['sin_phi', 'sin_psi', 'sin_omega', 'cos_phi', 'cos_psi', 'cos_omega', 'env', 'rmsd', 'term_len']),  #
}

DEFAULT_TRAIN_HPARAMS = {
    'term_matches_cutoff': None,
    # 'test_term_matches_cutoff': None,
    # ^ is an optional hparam if you want to use a different TERM matches cutoff during validation/testing vs training
    'train_batch_size': 16,
    'shuffle': True,
    'sort_data': True,
    'semi_shuffle': False,
    'regularization': 0,
    'max_term_res': 55000,
    'max_seq_tokens': None,
    'term_dropout': None,
    'loss_config': {
        'nlcpl': 1
    },
    'finetune': False,
    'finetune_lr': 1e-6
}
