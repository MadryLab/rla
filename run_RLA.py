# code to run RLA on wds input files

import src.models_and_optimizers as model_utils
import yaml
from types import SimpleNamespace
from clip_main import get_wds_loaders
from transformers import EsmTokenizer
import src.data_utils as data_utils
import os
import torch
import sys
sys.path.insert(0, '/data1/groups/keatinglabs/rla_shared')
import pickle
from tqdm import tqdm
import numpy as np
from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align
import matplotlib.pyplot as plt
import json
from torch.cuda.amp import autocast
import tmscoring
import json
import copy
from scipy.stats.stats import pearsonr
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, PrecisionRecallDisplay, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import pandas as pd
import glob
import webdataset as wds
sys.path.insert(0, '/data1/groups/keatinglab/tools')
from run_dockq import run_dockq
from transformers import EsmTokenizer, EsmModel
import esm as esmlib

#######

## GENERAL SETUP (NO CHANGES NEEDED)
ROOT = "/data1/groups/keatinglab/rla_shared/runs/new_blacklist"
#ROOT = "/data1/groups/keating_madry/runs/new_blacklist"
model_dir = "version_0/" 
dev = 'cuda:0'
CLIP_MODE = False
root_path = os.path.join(ROOT, model_dir)
path = os.path.join(root_path, "checkpoints/checkpoint_best.pt")
data_root = "/data1/groups/keatinglab/rla_shared/wds" #
args_path = os.path.join(ROOT, model_dir, [u for u in os.listdir(os.path.join(ROOT, model_dir)) if u.endswith('.pt')][0])

backwards_compat = {
    'masked_rate': -1,
    'masked_mode': 'MASK',
    'lm_only_text': 1,
    'lm_weight': 1,
    'resid_weight': 1,
    'language_head': False,
    'language_head_type': 'MLP',
    'zip_enabled': False,
    'num_mutations': False,
}
hparams = torch.load(args_path)
args_dict = hparams['args']
args_dict['data_root'] = data_root
args_dict['train_wds_path'] = 'wds_'
args_dict['val_wds_path'] = 'wds_'
args_dict['batch_size'] = 1
args_dict['blacklist_file'] = ''
for k in backwards_compat.keys():
    if k not in args_dict:
        args_dict[k] = backwards_compat[k]
args = SimpleNamespace(**args_dict)

print(vars(args))

coordinator_params = data_utils.get_coordinator_params(args.coordinator_hparams)
coordinator_params['num_positional_embeddings'] = args.gnn_num_pos_embs
coordinator_params['zero_out_pos_embs']= args.gnn_zero_out_pos_embs
coordinator_params['clip_mode'] = True

##########

## LOAD MODEL (NO CHANGES NEEDED)
args_dict['arch']= '/data1/groups/keatinglab/rla_shared/esm_model_150'
trained_model = model_utils.load_model(path, args_dict['arch'], dev)
tokenizer = EsmTokenizer.from_pretrained(args_dict['arch'])
esm_arch = '/data1/groups/keatinglab/rla_shared/esm_model_150'
esm_model = EsmModel.from_pretrained(args_dict['arch']) 
esm_model = esm_model.to(dev)
esm_model.eval()
esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
# esm, alphabet = esmlib.pretrained.esm1v_t33_650M_UR90S_1()
esm = esm.eval()
if dev == 'cuda:0':
    esm = esm.cuda()

#########

## LOAD UTIL FUNCTIONS (NO CHANGES NEEDED)
##%run 2023_12_11_rla_utils.ipynb

## edited to load in functions in a .py
from rla_utils_2023_12_11 import *
trained_model = trained_model.eval()

########

## DEFINE RLA SETTING CALCULATIONS (SOME CHANGES POSSIBLE)
seq_mask='peptide' # 'peptide' to mask peptide sequence, 'protein' to mask protein sequence
struct_mask=None # 'peptide' to mask peptide structure, 'protein' to mask protein structure
top_k = 30 # Num neighbors, probably want to keep at 30 but can experiment
focus = True # RLA calculation setting that limits RLA score to interface, almost certainly want to keep True
remove_far = True # Removes residues too far from the interface, likely want to keep True
threshold = 1 # Threshold for remove_far calculation, likely want to keep at 1
weight_dists = False # Weights RLA score per residue by distance from interface, likely want to keep False
pep_weight = 1 # Weight of peptide residues relative to protein residues, likely want to keep at 1

########

## DATASETS TO EVALUATE (CHANGE AS NEEDED)
name = sys.argv[1]
design_sets = ["rf_"+name]

########

## PERFORMS RLA CALCULATION (NO CHANGES NEEDED)
if CLIP_MODE:
    feature_getter = get_text_and_image_features_clip
else:
    feature_getter = get_text_and_image_features
    
torch.multiprocessing.set_sharing_strategy('file_system')
nclash_dict, Fnat_dict, Fnonnat_dict, LRMS_dict, iRMSDbb_dict, irmsdsc_dict, distance_dict, theta_dict, class_dict = {}, {}, {}, {}, {}, {}, {}, {}, {}
dicts = [nclash_dict, Fnat_dict, Fnonnat_dict, LRMS_dict, iRMSDbb_dict, irmsdsc_dict, distance_dict, theta_dict, class_dict]
nclash_data, Fnat_data, Fnonnat_data, LRMS_data, iRMSDbb_data, irmsdsc_data, distance_data, theta_data, class_data = {}, {}, {}, {}, {}, {}, {}, {}, {}
data_dicts = [nclash_data, Fnat_data, Fnonnat_data, LRMS_data, iRMSDbb_data, irmsdsc_data, distance_data, theta_data, class_data]
args.batch_size = 1
args.zip_enabled = False
args.num_mutations = 0
args.distributed = 0
plot_scores = []
plot_weights = []
plot_pep_mask = []
plot_indices = []
plot_X = []
plot_seq = []
paired_res_scores = {}
scores_stats = {'models': [], 'seqs': [], 'rla_scores': []}
# result_types = ['nclash', 'fnat', 'fnonnat', 'lrmsd', 'irmsdbb', 'irmsdsc', 'distance', 'theta', 'classification']
for design_set in design_sets:
    print(f'running on {design_set}.')
    args.train_wds_path = f"{design_set}.wds"
    args.val_wds_path = f"{design_set}.wds"
    train_loader, val_loader, train_len, val_len = get_wds_loaders(args, coordinator_params, gpu=None, shuffle_train=False, val_only=True, return_count=False)
    lens = {}
    for i, b in tqdm(enumerate(val_loader), total=val_len):
        lens[b[0]['pdb_id'][0]] = len(b[0]['string_sequence'][0])
    MAX_LEN = max(lens.values())
    
    for i, batch in enumerate(tqdm(val_loader, total=val_len)):
        model = batch[0]['pdb_id'][0]
        pep_seq = batch[0]['string_sequence'][0][:batch[1]['seq_lens'][0][0]]
        chain_lens = torch.zeros(batch[1]['coords'][0].shape[1]).to(device = batch[1]['coords'][0].device)
        chain_lens[batch[1]['seq_lens'][0][0]:] = 1
        chain_lens_mask = torch.ones(batch[1]['coords'][0].shape[1]).unsqueeze(0).to(dtype=torch.bool, device = batch[1]['coords'][0].device)
        batch[1]['chain_lens'] = [chain_lens.unsqueeze(0), chain_lens_mask]
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                output_dict = feature_getter(trained_model, tokenizer, batch, pdb=None, weight_dists=weight_dists, seq_mask=seq_mask, focus=focus, top_k=top_k, struct_mask=struct_mask, remove_far=remove_far, threshold=threshold, dev=dev)
                score, scores, plot_scores, plot_weights, plot_pep_mask, plot_indices, plot_X, plot_seq = compute_score(batch, output_dict, weight_dists, MAX_LEN, plot_scores=plot_scores, plot_weights=plot_weights, plot_pep_mask=plot_pep_mask, plot_indices=plot_indices, plot_X=plot_X, plot_seq=plot_seq, is_complex=True)
                scores_stats['models'].append(model)
                scores_stats['seqs'].append(pep_seq)
                paired_res_scores[model] = score

###########

## SAVE/LOAD RAW SCORES AND SEQUENCE MAPPINGS (CHANGE JSON WRITING/READING AS NEEDED)
path = "/data1/groups/keatinglab/rla_shared/relb_results/"

with open(path+'denovo_seq_mapping.json', 'w') as f:
    json.dump(scores_stats, f)
with open(path+'denovo_all_score_values_trim.json', 'w') as f:
    json.dump(paired_res_scores, f)

##########

## CONVERT RLA SCORES TO DF (CHANGE AS NEEDED)
rla_stats = {'models': [], 'sequence': [], 'RLA': []}
for k in paired_res_scores.keys():
    rla_stats['models'].append(k)
    rla_stats['RLA'].append(paired_res_scores[k])
    rla_stats['sequence'].append(scores_stats['seqs'][scores_stats['models'].index(k)])
rla_df = pd.DataFrame(rla_stats)
rla_extend_df = rla_df[rla_df['models'].str.contains('extension')]

rla_df.to_csv("/home/gridsan/dbritton/rele_Repeat_jgan/rele_binder_design/"+name+"/"+name+"_rla_scores.csv")

###########
