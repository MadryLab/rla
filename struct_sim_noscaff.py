import numpy as np
from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align
import tmscoring
from clip_main import get_wds_loaders
import src.data_utils as data_utils
from types import SimpleNamespace
import os
import torch
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import multiprocessing as mp
import pickle

# Function to save array to PDB format
def save_to_pdb(array, filename):
    with open(filename, 'w') as f:
        for i, residue in enumerate(array):
            for j, atom in enumerate(residue):
                f.write(f"ATOM  {j+1:>5}  CA  ALA A{i+1:>4}    {atom[0]:>8.3f}{atom[1]:>8.3f}{atom[2]:>8.3f}  1.00  0.00           C\n")
                
# Function to save array to PDB format
def save_to_pdb_ca(array, filename):
    with open(filename, 'w') as f:
        for j, atom in enumerate(array):
            f.write(f"ATOM  {j+1:>5}  CA  ALA A{j+1:>4}    {atom[0]:>8.3f}{atom[1]:>8.3f}{atom[2]:>8.3f}  1.00  0.00           C\n")

def calc_dist_matrix(struct_stats, target):
    dist_matrix = np.zeros((len(struct_stats), len(struct_stats)), dtype=np.float32)
    dist_matrix[dist_matrix == 0] = np.nan
    clusts = {}
    inds_in_clusts = set()
    for Aind, (A, Aseq, Aname) in enumerate(struct_stats):
        if Aind in inds_in_clusts:
            continue
        if Aind > 0 and ((not np.isnan(dist_matrix[:Aind,Aind]).all()) and np.nanmax(dist_matrix[:Aind, Aind]) > 0.8):
            max_ind = np.argmax(dist_matrix[:Aind, Aind])
            dist_matrix[Aind, :] = dist_matrix[max_ind, :]
            # dist_matrix[:, Aind] = dist_matrix[:, max_ind]
            continue
        diff_test = np.abs(dist_matrix[:Aind, :] - dist_matrix[:Aind, Aind][:, np.newaxis])
        for Bind in range(Aind, len(struct_stats)):
            if Bind in inds_in_clusts:
                continue
            B, Bseq, Bname = struct_stats[Bind]
            if Bind == Aind:
                dist_matrix[Aind, Bind] = 1
                continue
            if Aind > 0:
                if (not np.isnan(diff_test[:,Bind]).all()) and np.nanmax(diff_test[:,Bind]) > 0.25:
                    continue
            out = tm_align(A, B, Aseq, Bseq)
            tm = np.mean([out.tm_norm_chain1, out.tm_norm_chain2])
            dist_matrix[Aind, Bind] = tm
            dist_matrix[Bind, Aind] = tm
        clust_ind = np.where(dist_matrix[Aind] > 0.8)
        clust_ind = clust_ind[0]
        inds_in_clusts.update(list(clust_ind))
        clusts[len(clusts.keys())] = list(clust_ind)

    print(f"Done with {target} | {len(dist_matrix)}!")
    
    np.save(f'/home/gridsan/fbirnbaum/joint-protein-embs/relB_results/{target}_{scaff}_dist_matrix_strict.npy', dist_matrix)

    with open(f'/home/gridsan/fbirnbaum/joint-protein-embs/relB_results/{target}_{scaff}_clusts_strict.pickle', 'wb') as f:
        pickle.dump(clusts, f)

    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run downstream analysis')
    parser.add_argument('--my_task_id', help='Task id for batch array', required=True, type=int)
    parser.add_argument('--num_tasks', help='Number of jobs running simultaneously in batch array', required=True, type=int)
    run_args = parser.parse_args()
    print("my task id: ", run_args.my_task_id, " num tasks: ", run_args.num_tasks)
    model_dir = "version_0/" 
    #model_dir = "6_1_pdb/version_0" 

    CLIP_MODE = False
    dev='cpu'

    # need to edit these paths for supercloud  keating_madry_shared/models/9_17_big_coordinator
    ROOT = "/data1/groups/keating_madry/runs/new_blacklist"

    root_path = os.path.join(ROOT, model_dir)
    path = os.path.join(root_path, "checkpoints/checkpoint_best.pt")
    data_root = "/data1/groups/keating_madry/wds/" #
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
    args_dict['train_wds_path'] = 'baker_InsulinR_final.wds'
    args_dict['val_wds_path'] = 'baker_InsulinR_final.wds'
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

    file = '/data1/groups/keating_madry/baker_designed_binders/all_data/retrospective_analysis/af2_rmsd_graphs1_data.sc'
    df = pd.read_csv(file, sep=' ')

    # targets = ['FGFR2', 'H3', 'IL7Ra', 'InsulinR', 'PDGFR', 'SARS_CoV2_RBD', 'EGFR', 'VirB8']
    targets = ['denovo_2seed_coordinator_fused_allstructures', 
              'denovo_2seed_mpnn_fused_allstructures',
              'denovo_3seed_coordinator_fused_allstructures',
              'denovo_3seed_mpnn_fused_allstructures',
              'relBextension_2seed_coordinator_fused_allstructures']

    my_targets = targets[run_args.my_task_id:len(targets):run_args.num_tasks]
    print("My targets: ", my_targets)

    for target in my_targets:
        df_target = df[df['target'].str.contains(target)]
        # args.data_root = '/data1/groups/keating_madry/wds/'
        args.data_root = '/data1/groups/keatinglab/relE_binder_design'
        args.train_wds_path = f'{target}.wds'
        args.val_wds_path = f'{target}.wds'

        train_loader, val_loader, train_len, val_len = get_wds_loaders(args, coordinator_params, gpu=None, shuffle_train=False, val_only=True, return_count=False)
        
        
        file = f'/home/gridsan/fbirnbaum/joint-protein-embs/relB_results/{target}_peptide_struct_stats.json'
        if os.path.exists(file):
            with open(file, 'r') as f:
                struct_stats_save = json.load(f)
            struct_stats = []
            for ss in struct_stats_save:
                struct_stats.append((np.array(ss[0]), ss[1], ss[2]))
        else:
            struct_stats = []
            for i, b in tqdm(enumerate(val_loader), total=val_len):
                A = b[1]['coords'][0].numpy()[0].astype(np.float64)[:,1]
                A = A[:b[1]['seq_lens'][0][0]]
                Aseq = np.array(list(b[0]['string_sequence'][0]))
                Aseq = "".join(Aseq[b[0]['placeholder_mask'][0][0].numpy()][:b[1]['seq_lens'][0][0]])
                name = b[0]['pdb_id'][0]
                struct_stats.append((A, Aseq, name))

            struct_stats_save = []
            for ss in struct_stats:
                struct_stats_save.append((ss[0].tolist(), ss[1], ss[2]))


            with open(f'/home/gridsan/fbirnbaum/joint-protein-embs/relB_results/{target}_peptide_struct_stats.json', 'w') as f:
                json.dump(struct_stats_save, f)
        # struct_comps = np.zeros((val_len, val_len))
        # for i, b in tqdm(enumerate(struct_stats)):
        #     A, Aseq, Aname = b
        #     for j, d in enumerate(struct_stats):
        #         if j < i:
        #             continue
        #         if j == i:
        #             struct_comps[i, i] = 1
        #             continue
        #         B, Bseq, Bname = d
        #         out = tm_align(A, B, Aseq, Bseq)
        #         struct_comps[i, j] = np.mean([out.tm_norm_chain1, out.tm_norm_chain2])
        #         struct_comps[j, i] = struct_comps[i, j]
                
            # np.save(f'/home/gridsan/fbirnbaum/joint-protein-embs/baker_results/{target}_peptide_struct_comps.npy', struct_comps)


        print("Done getting struct stats")

        scaff_class_stats = {}
        scaff_classes = list(set(df_target['scaff_class'].values))

        for b in tqdm(struct_stats):
            name = b[2]
            if name not in df_target['description'].values:
                continue
            scaff = df_target[df_target['description'] == name]['scaff_class'].values[0]
            if scaff not in scaff_class_stats.keys():
                scaff_class_stats[scaff] = {}
            scaff_class_stats[scaff][name] = b


        calc_dist_matrix(struct_stats)

        print('Done')
                


