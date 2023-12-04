""" Functions to parse :code:`.etab` files."""
import argparse
import pickle

import numpy as np

from terminator.utils.common import AA_to_int


def parseEtab(filename, save=True):
    """Given an etab file, parse the corresponding Potts model parameters.

    This assumes that full chain protein etabs are provided, so it is not designed to
    work with etabs computed on partial chains.

    Args
    ====
    filename : str
        path to :code:`.etab` file
    save ; bool, default=True
        whether or not to save the outputs to a file

    Returns
    =======
    potts_dict : dict
        Sparse representation of etab that maps pairs of
        (residue1_id : int, residue2_id : int) to 22 x 22 matrices representing
        pair energies. Self energies are incorperated into the diagonals of
        self-interaction matricies, with key (residue_id, residue_id). The dimension is 22 because
        0 is used as a padding index and 21 is used to represent X for unknown residues.
    potts_selfE : np.ndarray
        Numpy array specifying self energies.
        Shape: all_chains_length x 22.
    potts : np.ndarray
        Dense representation of potts model mapping pairs of residue ids
        (the first two dimensions) to 22 x 22 matricies representing pair
        energies. Self energies are incorperated into the diagonals of
        self-interactions matricies. Edges are also duplicated e.g. interaction energies
        between residues 1 and 2 can be found at both :code`potts[1][2]`
        and :code:`potts[2][1]`.
        Shape: all_chains_length x all_chains_length x 22 x 22

    """

    if filename.split('.')[-1] != 'etab':
        raise ValueError('Input file is not an etab file!')

    selfE = []
    pairE = []
    id_to_resid = {}
    fp = open(filename, 'r')
    # this loop requires that all self energies
    # occur before pair energies
    for idx, line in enumerate(fp):
        l = line.strip().split(' ')

        # self energies
        if len(l) == 3:
            id = l[0]
            resid = idx // 20
            id_to_resid[id] = resid

            residue = AA_to_int[l[1]]
            E = float(l[2])

            selfE.append({'resid': resid, 'residue': residue, 'E': E})

        # couplings
        elif len(l) == 5:
            id0 = l[0]
            id1 = l[1]
            resid0 = id_to_resid[id0]
            resid1 = id_to_resid[id1]

            residue0 = AA_to_int[l[2]]
            residue1 = AA_to_int[l[3]]
            E = float(l[4])

            pairE.append({'resid0': resid0, 'resid1': resid1, 'residue0': residue0, 'residue1': residue1, 'E': E})
        else:
            raise ValueError("Something doesn't look right at line %d: %s" % (idx, line))

    fp.close()

    # number of amino acids in all chains
    L = max(id_to_resid.values()) + 1

    potts_selfE = np.zeros((L, 22))
    potts = np.zeros((L, L, 22, 22))

    potts_dict = {}

    for data in selfE:
        resid = data['resid']
        residue = data['residue']

        # self E
        potts_selfE[resid][residue] = data['E']
        # dense potts
        slice = potts[resid][resid]
        slice[residue][residue] = data['E']
        # sparse potts
        key = (resid, resid)
        idx = residue * 22 + residue
        if key not in potts_dict.keys():
            potts_dict[key] = np.zeros(22 * 22)
        potts_dict[key][idx] = data['E']

    for data in pairE:
        resid0 = data['resid0']
        resid1 = data['resid1']
        residue0 = data['residue0']
        residue1 = data['residue1']
        # dense potts
        slice0 = potts[resid0][resid1]
        slice0[residue0][residue1] = data['E']
        slice1 = potts[resid1][resid0]
        slice1[residue1][residue0] = data['E']

        # sparse potts
        key1 = (resid0, resid1)
        idx1 = residue0 * 22 + residue1
        if key1 not in potts_dict.keys():
            potts_dict[key1] = np.zeros(22 * 22)
        potts_dict[key1][idx1] = data['E']

        key2 = (resid1, resid0)
        idx2 = residue1 * 22 + residue0
        if key2 not in potts_dict.keys():
            potts_dict[key2] = np.zeros(22 * 22)
        potts_dict[key2][idx2] = data['E']

    if save:
        np.save(filename, potts)
        np.save(filename[:-5] + '_selfE.etab', potts_selfE)
        with open('potts_dict.etab') as fp:
            pickle.dump(potts_dict, fp)

    # testing that the values in the potts parameters is correct
    """
    np.set_printoptions(threshold=np.inf, precision=2, linewidth=300)
    print(potts[0][0])
    """

    return potts_dict, potts_selfE, potts
