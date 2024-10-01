import torch
import h5py
import numpy as np
import sys
import webdataset as wds
import os
import copy
import tqdm
from torch.utils.data import DataLoader
import importlib
import clip_main
import src.loader as loaders_utils
import warnings
import gzip
import random
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser
from Bio import SeqIO
from Bio.SeqUtils import seq1
import pandas as pd
import tarfile
warnings.simplefilter('ignore', PDBConstructionWarning)

def process_residue(residue):
    atoms = ['N', 'CA', 'C', 'O']
    coordinates = []
    for r in atoms:
        coord = residue.child_dict.get(r, None)
        if coord is None:
            if r == 'O':
                coord = residue.child_dict.get('OXT', None)
            if coord is None:
                return None, None
        coordinates.append(np.array(coord.get_coord()))
    return np.stack(coordinates), seq1(residue.resname)

def process_chain(chain):
    coordinates = []
    seq = []
    for r in chain:
        output, residue_name = process_residue(r)
        if output is not None:
            coordinates.append(output)
            seq.append(residue_name)
    if len(coordinates) == 0:
        return None
    coordinates = np.stack(coordinates)
    seq = ''.join(seq)
    return coordinates, seq

def process_chains(chains, pep=False, prot=False):
    if pep or prot:
        chain_lens = []
        chain_ids = []
        for chain in chains:
            for i, res in enumerate(chain):
                continue
            chain_lens.append(i)
            chain_ids.append(chain.id)
        if chain_lens[0] < chain_lens[1]:
            pep_id = chain_ids[0]
            prot_id = chain_ids[1]
        else:
            pep_id = chain_ids[1]
            prot_id = chain_ids[0]
        if pep and isinstance(pep, str): pep_id == pep
        if prot and isinstance(prot, str): prot_id == prot
    output = []
    chain_ids = []
    for chain in chains:
        if (pep and chain.id != pep_id) or (prot and chain.id != prot_id):
            continue
        out = process_chain(chain)
        if out is not None:
            output.append(out)
            chain_ids.append(chain.id)
    coords = [u[0] for u in output]
    seqs = [u[1] for u in output]
    return coords, seqs, chain_ids

def process_structure(structure, pep=False, prot=False):
    for s in structure: # only one structure
        return process_chains(s, pep, prot)
    return None

# +
def process_pdb(parser, pdb_filename):
    # print(pdb_filename)
    with gzip.open(pdb_filename, "rt") as file_handle:
        structure = parser.get_structure("?", file_handle)
        date = structure.header['deposition_date']
        return process_structure(structure), date
    
def process_pdb_raw(parser, pdb_filename, pep=False, prot=False):
    s = parser.get_structure("?", pdb_filename)
    return process_structure(s, pep, prot)

def read_input_ids(index_file):
    input_ids = []
    with open(os.path.join(index_file), 'r') as f:
        for line in f:
            input_ids += [line.strip()]
    return np.array(input_ids)

def write_dataset(dataset, tar_name, use_shards=False, max_shard_count=10000):
    if use_shards:
        os.makedirs(tar_name, exist_ok=True)
        sink = wds.ShardWriter(f'{tar_name}/shard-%06d.tar',maxcount=max_shard_count)
    else:
        sink = wds.TarWriter(tar_name)
    for index, (batch, pdb_id) in enumerate(dataset):
        if index%1000==0:
            print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
        if len(batch[0]) == 0:
            continue
        sink.write({
            "__key__": "sample%06d" % index,
            "inp.pyd": dict(coords=batch[0], seqs=batch[1], chain_ids=batch[2], pdb_id=pdb_id),
        })
    sink.close()