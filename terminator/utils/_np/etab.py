import numpy as np

from terminator.utils.common import seq_to_ints


def knnEtab_to_denseEtab(h_E, E_idx, pad=True):
    """
    Convert a kNN etab to a dense etab
    """
    n_nodes, k, n_aa, _ = h_E.shape
    # collect edges into NxN tensor shape
    collection = np.zeros((n_nodes, n_nodes, n_aa, n_aa))
    neighbor_idx = np.broadcast_to(E_idx[..., np.newaxis, np.newaxis], (n_nodes, k, n_aa, n_aa))
    np.put_along_axis(collection, neighbor_idx, h_E, 1)
    if pad:
        collection = np.pad(collection, ((0, 0), (0, 0), (0, 2), (0, 2)))
    return collection


def eval_seq_energy(etab, seq):
    """
    Evaluate the energy of a sequence given a dense etab
    """
    if isinstance(seq, str):
        seq = seq_to_ints(seq)
        seq = np.array(seq)

    seq_len = len(seq)
    energy = 0
    for i in range(seq_len):
        r_i = seq[i]
        for j in range(seq_len):
            r_j = seq[j]
            if i > j:  # we don't want to double count pair energies
                continue
            energy += etab[i][j][r_i][r_j]

    return energy
