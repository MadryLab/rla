# script to convert pdbs into wds , readable for RLA

from WDS_functions import *
import glob
import sys

name = sys.argv[1]
dir_ = "/home/gridsan/jgan/rele_binder_design/"+name+"/backbones/"
parser = PDBParser()
root_pdb = dir_
outputs = []

for i, pdb_file in tqdm.tqdm(enumerate(glob.glob(dir_+"*.pdb")), total=len(glob.glob(dir_+"*.pdb"))):
#for i, pdb_file in tqdm.tqdm(enumerate(os.listdir(dir_)), total=len(os.listdir(dir_))):
    pdb_file = pdb_file.strip()
    pdb_file = os.path.join(dir_, pdb_file)
    out = process_pdb_raw(parser, pdb_file)
    pdb_id = pdb_file.split('.')[0]
    outputs.append((out, pdb_id))
    # os.remove(pdb_filename)

output_dicts = {'train': []} #{'train': [], 'test': [], 'val': []}
for o, pdb_id in tqdm.tqdm(outputs):
    if o is None:
        continue
    output_dicts['train'].append((o, pdb_id))

for k, dataset in output_dicts.items():
    if k == 'train':
        tar_name = "/home/gridsan/jgan/keatinglab_shared/rla_shared/wds/rf_"+name+".wds"
        write_dataset(dataset, tar_name)
