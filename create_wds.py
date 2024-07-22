import torch
import h5py
import numpy as np
import sys
sys.path.insert(0, '/data1/groups/keating_madry')
sys.path.insert(0, '/data1/groups/keatinglab/relE_binder_design')
import webdataset as wds
import os
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
import importlib
import clip_main
import src.loader as loaders_utils
import warnings
import gzip
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser
from Bio import SeqIO
from Bio.SeqUtils import seq1
import pandas as pd
import tarfile
warnings.simplefilter('ignore', PDBConstructionWarning)
import argparse
import random

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
            "inp.pyd": dict(coords=batch[0][0], seqs=batch[0][1], chain_ids=batch[0][2], pdb_id=os.path.splitext(os.path.splitext(os.path.basename(pdb_id))[0])[0]),
        })
    sink.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train converter!')
    parser.add_argument('--out_dir', help='Directory to save training results', default='/data1/groups/keating_madry/wds')
    parser.add_argument('--in_dir', help='Raw data', default='/data1/groups/keating_madry/PDB/pdb')
    parser.add_argument('--blacklist', help='PDB ids to exclude from train set', default='/data1/groups/keating_madry/mlsb_pdb_blacklist_v2.txt')
    parser.add_argument('--date_cut', help='Date cutoff defining train/val split', default='2021-08-01')
    args = parser.parse_args()

    print(f'Out dir: {args.out_dir}')
    print(f'In dir: {args.in_dir}')
    print(f'Blacklist: {args.blacklist}')
    print(f'Date cut: {args.date_cut}')

    parser = PDBParser()
    with open(args.blacklist, 'r') as f:
        blacklist = f.readlines()
    blacklist = [blac.strip() for blac in blacklist]

    output_dicts = {'train': [], 'val': [], 'test': []}
    for subdir in tqdm(os.listdir(args.in_dir)):
        subdir = subdir.strip()
        subdir = os.path.join(args.in_dir, subdir)
        if not os.path.isdir(subdir):
            continue
        for pdb_file in os.listdir(subdir):
            pdb_id = pdb_file.split('.')[0][3:]
            pdb_file = pdb_file.strip()
            pdb_file = os.path.join(subdir, pdb_file)
            try:
                out = process_pdb(parser, pdb_file)
                if pdb_id in blacklist:
                    output_dicts['test'].append((out, pdb_id))
                elif out[-1] > args.date_cut:
                    if random.random() > 0.5:
                        output_dicts['test'].append((out, pdb_id))
                    else:
                        output_dicts['val'].append((out, pdb_id))
                elif len(out[0][0]) == 0:
                    continue
                else:
                    output_dicts['train'].append((out, pdb_id))
            except Exception as e:
                continue
        

    train_out = os.path.join(args.out_dir, 'full_pdb_train.wds')
    val_out = os.path.join(args.out_dir, 'full_pdb_val.wds')
    test_out = os.path.join(args.out_dir, 'full_pdb_test.wds')

    write_dataset(output_dicts['train'], train_out)
    print('Done with train!')
    write_dataset(output_dicts['val'], val_out)
    print('Done with val')
    write_dataset(output_dicts['test'], test_out)
    print('Done with test')


    

        

