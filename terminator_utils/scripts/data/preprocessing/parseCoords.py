"""Functions to parse :code:`.red.pdb` files"""
import pickle
import os
import numpy as np
from terminator.utils.common import aa_three_to_one


def parseCoords(filename, save=True):
    """ Parse coordinates from :code:`.red.pdb` or :code:`.pdb` files, and dump in
    files if specified.

    Args
    ====
    filename : str
        path to :code:`.red.pdb` file or :code:`.pdb`

    save : bool, default=True
        whether or not to dump the results

    Returns
    =======
    chain_tensors : dict
        Dictionary mapping chain IDs to arrays of atomic coordinates.

    seq : str
        Sequence of all chains concatenated.

    res_info : list
        a '_' separated string (i.e. chainID_residueNumber) for each residue in the structure
    """
    chain_dict = {}
    from_raw_pdb = len(os.path.splitext(os.path.splitext(filename)[0])[1]) == 0
    VALID_ELEMENTS = ['N', 'CA', 'C', 'O', 'OXT']
    VALID_RECORD_TYPES = ['ATOM', 'HETATM']
    with open(filename, 'r') as fp:
        element_coords = {atom: [] for atom in ['N', 'CA', 'C', 'O']}
        cur_residx = 'NaN'
        for line in fp:
            data = line.strip()
            if data[:3] == 'TER' or data[:3] == 'END':
                continue
            record_type = data[0:6].strip()
            if from_raw_pdb and record_type not in VALID_RECORD_TYPES:
                continue
            try:
                element = data[13:16].strip()
                residue = data[17:20].strip()
                if residue == 'MSE':  # convert MSE (seleno-met) to MET
                    residue = 'MET'
                elif residue == 'SEP':  # convert SEP (phospho-ser) to SER
                    residue = 'SER'
                elif residue == 'TPO':  # convert TPO (phospho-thr) to THR
                    residue = 'THR'
                elif residue == 'PTR':  # convert PTR (phospho-tyr) to TYR
                    residue = 'TYR'
                elif residue == 'CSO':  # convert CSO (hydroxy-cys) to CYS
                    residue = 'CYS'
                elif residue == 'SEC':  # convert SEC (seleno-cys) to CYS
                    residue = 'CYS'
                residx = data[22:27].strip()
                chain = data[21]
                x = data[30:38].strip()
                y = data[38:46].strip()
                z = data[46:54].strip()
                coords = [float(coord) for coord in [x, y, z]]
                # print(element, chain, coords)
            except Exception as e:
                print(data)
                raise e

            if element not in VALID_ELEMENTS:
                continue
            # naively model terminal carboxylate as a single O atom
            # (i cant find the two oxygens so im just gonna use OXT)
            if element == 'OXT':
                element = 'O'
            
            if (not cur_residx == 'NaN') and (residx != cur_residx):
                element_coords = {atom: [] for atom in ['N', 'CA', 'C', 'O']}
                cur_residx = residx
                element_coords[element] = coords
                continue
            
            cur_residx = residx
            element_coords[element] = coords
            seen_all_atoms = all([len(element_coords[atom]) > 0 for atom in element_coords.keys()])
            if seen_all_atoms:
                for element_id, coords in zip(element_coords.keys(), element_coords.values()):
                    if chain not in chain_dict.keys():
                        chain_dict[chain] = {atom: [] for atom in ['N', 'CA', 'C', 'O']}
                        chain_dict[chain]["seq_dict"] = {}
                        chain_dict[chain]["res_info"] = []

                    chain_dict[chain][element_id].append(coords)

                    seq_dict = chain_dict[chain]["seq_dict"]
                    if residx not in seq_dict.keys():
                        seq_dict[residx] = aa_three_to_one(residue)
                chain_dict[chain]["res_info"].append((chain,residx))
                element_coords = {atom: [] for atom in ['N', 'CA', 'C', 'O']}
            
    chain_tensors = {}
    seq = ""
    res_info = list()
    for chain in chain_dict.keys():
        coords = [chain_dict[chain][element] for element in ['N', 'CA', 'C', 'O']]
        chain_tensors[chain] = np.stack(coords, 1)
        seq_dict = chain_dict[chain]["seq_dict"]
        chain_seq = "".join([seq_dict[i] for i in seq_dict.keys()])
        chain_res_info = chain_dict[chain]["res_info"]
        assert len(chain_seq) == chain_tensors[chain].shape[0], (chain_seq, chain_tensors[chain].shape, filename)
        assert len(chain_res_info) == chain_tensors[chain].shape[0], (chain_res_info, chain_tensors[chain].shape, filename)
        seq += "".join([seq_dict[i] for i in seq_dict.keys()])
        res_info += chain_res_info

    if save:
        with open(filename[:-8] + '.coords', 'wb') as fp:
            pickle.dump(chain_tensors, fp)
        with open(filename[:-8] + '.seq', 'w') as fp:
            fp.write(seq)

    return chain_tensors, seq, res_info
