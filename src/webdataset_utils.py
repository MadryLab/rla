import numpy as np
import gzip
import warnings
from pathlib import Path
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser
from Bio import SeqIO
from Bio.SeqUtils import seq1
import webdataset as wds
import sys
import torch
import numpy as np
import os
import tqdm
import pandas as pd
from datetime import datetime
import argparse
import tarfile


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

def process_chains(chains):
    output = []
    chain_ids = []
    for chain in chains:
        out = process_chain(chain)
        if out is not None:
            output.append(out)
            chain_ids.append(chain.id)
    coords = [u[0] for u in output]
    seqs = [u[1] for u in output]
    return coords, seqs, chain_ids

def process_structure(structure):
    for s in structure: # only one structure
        return process_chains(s)
    return None

# +
def process_pdb(parser, pdb_filename):
    # print(pdb_filename)
    with gzip.open(pdb_filename, "rt") as file_handle:
        structure = parser.get_structure("?", file_handle)
        date = structure.header['deposition_date']
        return process_structure(structure), date
    
def process_pdb_raw(parser, pdb_filename):
    s = parser.get_structure("?", pdb_filename)
    return process_structure(s)


# -


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

def write_multichain_dataset():
    keating_madry_dir = "/home/gridsan/sajain/keating_madry_shared"
    parser = PDBParser()
    root_pdb = os.path.join(keating_madry_dir, "multichain_data/multichain_raw")
    pdb_files = os.listdir(root_pdb)
    outputs = []
    for i, pdb_file in tqdm.tqdm(enumerate(pdb_files)):
        pdb_filename = os.path.join(root_pdb, pdb_file)
        pdb_id = pdb_file.split('.')[0]
        out = process_pdb_raw(parser, pdb_filename)
        outputs.append((out, pdb_id))
    id_lists = {
        'train': os.path.join(keating_madry_dir, "multichain_data/train.in"),
        'val': os.path.join(keating_madry_dir, "multichain_data/validation.in"),
        'test': os.path.join(keating_madry_dir, "multichain_data/test.in"),
    }
    id_lists = {k: pd.read_csv(v, header=None)[0].to_numpy() for k,v in id_lists.items()}
    
    output_dicts = {'train': [], 'test': [], 'val': []}
    for o, pdb_id in tqdm.tqdm(outputs):
        if o is None:
            continue
        target_k = 'train'
        for k in id_lists.keys():
            if pdb_id in id_lists[k]:
                target_k = k
        output_dicts[target_k].append((o, pdb_id))
        
    for k, dataset in output_dicts.items():
        tar_name = f"multichain_clip_{k}.wds"
        write_dataset(dataset, tar_name)

def process_date_str(date_val):
    return datetime.strptime(date_val, "%Y-%m-%d")

def write_pdb_datasets(num_workers, worker_id, destination_folder):
    parser = PDBParser(QUIET=True)
    root_pdb = "/mnt/cfs/datasets/PDB/pdb/"
    pdb_dirs = [os.path.join(root_pdb, u) for u in os.listdir(root_pdb)]

    pdb_zips = []
    for pdb_dir in pdb_dirs:
        for u in os.listdir(pdb_dir):
            if u.endswith(".ent.gz"):
                pdb_zips.append(os.path.join(pdb_dir, u)) 
    pdb_zips = np.array(pdb_zips)[worker_id::num_workers]
                
    val_date_cutoff = process_date_str("2021-08-01")
    test_date_cutoff = process_date_str("2022-01-01")

    output_dict = {'train': [], 'test': [], 'val': []}            
    train_outputs, val_outputs, test_outputs = [], [], []
    for i, pdb_filename in enumerate(tqdm.tqdm(pdb_zips)):
        pdb_id = os.path.basename(pdb_filename).split('.')[0].split('pdb')[1]
        out, date = process_pdb(parser, pdb_filename)
        if out is None:
            continue
        date = process_date_str(date)
        if date > test_date_cutoff: # test set
            destination = 'test'
        elif date > val_date_cutoff: # val set
            destination = 'val'
        else: # train set
            destination = 'train'
        output_dict[destination].append((out, pdb_id))
    
    for k in output_dict.keys():
        os.makedirs(os.path.join(destination_folder, k), exist_ok=True)

    normalized_worker_id = "%06d" % worker_id
    train_tar_folder = os.path.join(destination_folder, f"train/{normalized_worker_id}")
    val_tar = os.path.join(destination_folder, f"val/{normalized_worker_id}.tar")
    test_tar = os.path.join(destination_folder, f"test/{normalized_worker_id}.tar")
    
    write_dataset(output_dict['train'], train_tar_folder, use_shards=True)
    write_dataset(output_dict['val'], val_tar, use_shards=False)
    write_dataset(output_dict['test'], test_tar, use_shards=False)

    
def write_alphafold_datasets(num_workers, worker_id, destination_folder):
    parser = PDBParser(QUIET=True)
    train_pdbs = []
    with tarfile.open("/mnt/xfs/projects/proteins/datasets/alphafold/swissprot_pdb_v4.tar", 'r') as file:
        names = file.getmembers()
        inds_to_do = np.arange(len(names))[worker_id::num_workers]
        for idx in tqdm.tqdm(inds_to_do):
            name = names[idx]
            f = file.extractfile(name)
            pdb_id = name.name.split('.pdb.gz')[0]
            with gzip.open(f, mode='rt') as f_gzip:
                output = process_pdb_raw(parser, f_gzip)
            if output is None:
                continue
            train_pdbs.append((output, pdb_id))
            f.close()

    normalized_worker_id = "%06d" % worker_id
    train_tar_folder = os.path.join(destination_folder, normalized_worker_id)
    write_dataset(train_pdbs, train_tar_folder, use_shards=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--destination-folder", type=str, default="pdb_webdataset")
    parser.add_argument("--dataset-type", type=str, default="PDB")
    args = parser.parse_args()
    print(args.num_workers, args.worker_id, args.destination_folder)
    if args.dataset_type == "PDB":
        write_pdb_datasets(args.num_workers, args.worker_id, args.destination_folder)
    elif args.dataset_type == "ALPHAFOLD":
        write_alphafold_datasets(args.num_workers, args.worker_id, args.destination_folder)
    else:
        assert False
        
    # parser = PDBParser()
    # root_pdb = "/mnt/cfs/datasets/PDB/pdb/"
    # pdb_dirs = [os.path.join(root_pdb, u) for u in os.listdir(root_pdb)]

    # pdb_zips = []
    # for pdb_dir in pdb_dirs:
    #     for u in os.listdir(pdb_dir):
    #         if u.endswith(".ent.gz"):
    #             pdb_zips.append(os.path.join(pdb_dir, u)) 
                
                
    # outputs = []
    # for i, pdb_filename in enumerate(tqdm.tqdm(pdb_zips)):
    #     pdb_id = os.path.basename(pdb_filename).split('.')[0].split('pdb')[1]
    #     out, date = process_pdb(parser, pdb_filename)
    #     outputs.append((out, pdb_id))
    #     if i > 20:
    #         break
