"""Functions that generate features from dTERMen files :code:`.dat` and :code:`.red.pdb`"""
import os
import pickle
import sys

import numpy as np

from terminator.utils.common import seq_to_ints

# for autosummary import purposes
sys.path.insert(0, os.path.dirname(__file__))
from parseCoords import parseCoords
from parseTERM import parseTERMdata

NUM_AA = 21  # including X
ZERO = 1e-10  # 0 is used for padding


def dumpTrainingTensors(in_path, out_path=None, cutoff=1000, save=True, coords_only=False, dummy_terms=None):
    """Generate features from dTERMen :code:`.dat` and :code:`.red.pdb`, and
    dump the output into a file if requested.

    Args
    ----
    in_path : str
        Prefix to :code:`.dat` and :code:`.red.pdb` files
    out_path : str or None
        Prefix to the output :code:`.features` and :code:`.length`. Can be None if :code:`save=False`.
    cutoff : int, default=1000
        Max number of TERMs to featurize
    save : bool, default=True
        Whether or not to save the training tensors
    coords_only : bool, default=False
        Whether to use only backbone-derived features
    dummy_terms : str or None
        Method by which to incorperate dummy TERMs. Options include :code:`'replace'`,
        which means replacing TERM features with those derived from a dummy TERM, or
        :code:`'include'`, which includes the dummy TERM into the mined TERM matches.

    Returns
    -------
    dict
        Dictionary of features for TERMinator
    """
    if dummy_terms is not None:
        assert dummy_terms in ['replace', 'include'], f"dummy_terms={dummy_terms} is an invalid argument"

    if dummy_terms == 'replace':
        cutoff = 1

    coords, _, _ = parseCoords(in_path + '.red.pdb', save=False)
    data = parseTERMdata(in_path + '.dat')
    # etab, self_etab, _ = parseEtab(in_path + '.etab', save=False)

    selection = data['selection']

    # embed target ppoe
    struct_ppoe = data['ppoe']
    struct_ppo = struct_ppoe[:, :3]
    struct_ppo_rads = np.radians(struct_ppo)
    struct_env = struct_ppoe[:, 3:]
    # zero out dihedral embeddings where there are no dihedrals
    struct_is_999 = (struct_ppo == 999)
    struct_sin_ppo = np.sin(struct_ppo_rads)
    struct_sin_ppo[struct_is_999] = 0
    struct_cos_ppo = np.cos(struct_ppo_rads)
    struct_cos_ppo[struct_is_999] = 0
    struct_embedded_ppoe = np.concatenate([struct_sin_ppo, struct_cos_ppo, struct_env], axis=1)

    term_msas = []
    term_features = []
    term_focuses = []
    term_contact_idxs = []
    term_lens = []
    # compute TERM features
    for term_data in data['terms']:
        focus = term_data['focus']
        # only take data for residues that are in the selection
        take = [i for i in range(len(focus)) if focus[i] in selection]

        msa = term_data['labels']
        # apply take
        msa = np.take(msa, take, axis=-1)
        if dummy_terms is None:
            # cutoff MSAs at top N
            term_msas.append(msa[:cutoff])
        elif dummy_terms == 'replace':
            # replace the whole TERM with one sequence of only X
            term_msas.append(np.ones_like(msa[:1]).astype(int) * 20)
        elif dummy_terms == "include":
            dummy_seq = np.ones_like(msa[:1]).astype(int) * 20
            term_msas.append(np.concatenate([dummy_seq, msa[:cutoff - 1]]))

        # add focus
        focus_take = [item for item in focus if item in selection]
        term_focuses += focus_take
        # add contact idx
        contact_idx = term_data['contact_idx']
        contact_idx_take = [contact_idx[i] for i in range(len(contact_idx)) if focus[i] in selection]
        term_contact_idxs += contact_idx_take
        # append term len, the len of the focus
        term_lens.append(len(focus_take))

        # process ppoe
        if dummy_terms is None:
            ppoe = term_data['ppoe']
        elif dummy_terms == "replace":
            ppoe = np.expand_dims(struct_ppoe[focus].transpose(1, 0), 0)
        elif dummy_terms == "include":
            dummy_ppoe = np.expand_dims(struct_ppoe[focus].transpose(1, 0), 0)
            ppoe = np.concatenate([dummy_ppoe, term_data['ppoe']], axis=0)
        term_len = ppoe.shape[2]
        num_alignments = ppoe.shape[0]
        # project to sin, cos
        ppo_rads = ppoe[:, :3] / 180 * np.pi
        is_999 = (ppoe[:, :3] == 999)
        sin_ppo = np.sin(ppo_rads)
        cos_ppo = np.cos(ppo_rads)
        # zero out dihedrals where there is no dihedral angle
        sin_ppo[is_999] = 0
        cos_ppo[is_999] = 0
        env = ppoe[:, 3:]

        # apply take
        ppoe = np.take(ppoe, take, axis=-1)

        # place rmsds into np array
        if dummy_terms is None:
            rmsd = np.expand_dims(term_data['rmsds'], 1)
        elif dummy_terms == "include":
            rmsd = np.concatenate([
                np.array([ZERO]),
                term_data['rmsds'],
            ], axis=0)
            rmsd = np.expand_dims(rmsd, 1)
        rmsd_arr = np.concatenate([rmsd for _ in range(term_len)], axis=1)
        rmsd_arr = np.expand_dims(rmsd_arr, 1)
        if dummy_terms == 'replace':
            # we set the RMSD of the true match to be 0
            rmsd_arr = np.ones_like(rmsd_arr) * ZERO
        term_len_arr = np.zeros((cutoff, 1, term_len))
        term_len_arr += term_len
        num_alignments_arr = np.zeros((cutoff, 1, term_len))
        num_alignments_arr += num_alignments

        # select features, cutoff at top N
        selected_features = [sin_ppo[:cutoff], cos_ppo[:cutoff], env[:cutoff], rmsd_arr[:cutoff], term_len_arr]

        features = np.concatenate(selected_features, axis=1)

        # pytorch does row vector computation
        # swap rows and columns
        features = features.transpose(0, 2, 1)
        term_features.append(features)

    msa_tensor = np.concatenate(term_msas, axis=-1)
    features_tensor = np.concatenate(term_features, axis=1)
    len_tensor = np.array(term_lens)
    term_focuses = np.array(term_focuses)
    term_contact_idxs = np.array(term_contact_idxs)

    # package cov matrices into one tensor
    max_term_len = max(term_lens)
    num_terms = len(term_lens)

    # check that sum of term lens is as long as the feature tensor
    assert sum(len_tensor) == features_tensor.shape[1]

    # manipulate coords to right shape
    pdb = in_path.split('/')[-1]

    coords_tensor = None
    if len(coords) == 1:
        chain = next(iter(coords.keys()))
        coords_tensor = coords[chain]
    else:
        chains = sorted(coords.keys())
        coords_tensor = np.vstack([coords[c] for c in chains])
    assert coords_tensor.shape[0] == len(data['sequence']), "num aa coords != seq length"

    output = {
        'pdb': pdb,
        'coords': coords_tensor,
        'ppoe': struct_embedded_ppoe,
        'features': features_tensor,
        'msas': msa_tensor,
        'focuses': term_focuses,
        'contact_idxs': term_contact_idxs,
        'term_lens': len_tensor,
        'sequence': np.array(data['sequence']),
        'seq_len': len(data['selection']),
        'chain_lens': data['chain_lens']
    }

    if coords_only:
        dummy_arr_3d = np.zeros([1, 1, 1])
        dummy_arr_2d = np.zeros([1, 1])
        output = {
            'pdb': pdb,
            'coords': coords_tensor,
            'ppoe': dummy_arr_3d,
            'features': dummy_arr_3d,
            'msas': dummy_arr_2d,
            'focuses': dummy_arr_2d,
            'contact_idxs': dummy_arr_2d,
            'term_lens': len_tensor,
            'sequence': np.array(data['sequence']),
            'seq_len': len(data['selection']),
            'chain_lens': data['chain_lens']
        }

    if save:
        assert out_path, "out_path required if save=True"

        with open(out_path + '.features', 'wb') as fp:
            pickle.dump(output, fp)
        with open(out_path + '.length', 'w') as fp:
            fp.write(str(len(term_focuses)) + '\n')
            fp.write(str(len(data['selection'])))

    print('Done with', pdb)
    return output


def dumpCoordsTensors(in_path, out_path=None, save=True):
    """Create a feature file based only on the coordinate information,
    placing dummy arrays for all TERM based items.

    Args
    ----
    in_path : str
        Prefix to :code:`.red.pdb` file
    out_path : str or None
        Prefix to the output :code:`.features` and :code:`.length`. Can be None if :code:`save=False`.
    save : bool, default=True
        Whether or not to save the training tensors

    Returns
    -------
    dict
        Dictionary of features for TERMinator
    """
    in_file = in_path + '.red.pdb'
    coords, seq, _ = parseCoords(in_file, save=False)

    if len(coords) == 1:
        chain = next(iter(coords.keys()))
        coords_tensor = coords[chain]
    else:
        chains = sorted(coords.keys())
        coords_tensor = np.vstack([coords[c] for c in chains])

    chain_lens = [len(coords[c]) for c in sorted(coords.keys())]
    pdb = os.path.basename(in_path)

    dummy_arr_3d = np.zeros([1, 1, 1])
    dummy_arr_2d = np.zeros([1, 1])
    dummy_arr_1d = np.ones([1])
    output = {
        'pdb': pdb,
        'coords': coords_tensor,
        'ppoe': dummy_arr_3d,
        'features': dummy_arr_3d,
        'msas': dummy_arr_2d,
        'focuses': dummy_arr_2d,
        'contact_idxs': dummy_arr_2d,
        'term_lens': dummy_arr_1d.astype(int),
        'sequence': np.array(seq_to_ints(seq)),
        'seq_len': len(seq),
        'chain_lens': chain_lens
    }

    if save:
        assert out_path, "out_path required if save=True"
        with open(out_path + '.features', 'wb') as fp:
            pickle.dump(output, fp)
        with open(out_path + '.length', 'w') as fp:
            fp.write(str(1) + '\n')
            fp.write(str(len(seq)))

    print('Done with', pdb)
    return output
