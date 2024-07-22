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


def calc_dist_matrix(prot_inter_sets, scaff_classes):
    all_inter_sets = []
    for scaff in scaff_classes:
        all_inter_sets += list(prot_inter_sets[scaff].values())

    prot_set_comps = np.ones((len(all_inter_sets), len(all_inter_sets)), dtype=np.float32)

    for i, iset in tqdm(enumerate(all_inter_sets), total=len(all_inter_sets)):
        for j in range(i+1, len(all_inter_sets)):
            jset = all_inter_sets[j]
            if len(set(iset).union(set(jset))) > 0:
                set_comp = len(set(iset).intersection(set(jset))) / len(set(iset).union(set(jset)))
            else:
                set_comp = np.nan
            prot_set_comps[i, j] = set_comp
            prot_set_comps[j, i] = set_comp

    np.save(f'/home/gridsan/fbirnbaum/joint-protein-embs/baker_results/binding_site/{target}_all_binding_site_comps.npy', prot_set_comps)

    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run downstream analysis')
    parser.add_argument('--my_task_id', help='Task id for batch array', required=True, type=int)
    parser.add_argument('--num_tasks', help='Number of jobs running simultaneously in batch array', required=True, type=int)
    run_args = parser.parse_args()
    print("my task id: ", run_args.my_task_id, " num tasks: ", run_args.num_tasks)

    file = '/data1/groups/keating_madry/baker_designed_binders/all_data/retrospective_analysis/af2_rmsd_graphs1_data.sc'
    df = pd.read_csv(file, sep=' ')
    targets = ['FGFR2', 'H3', 'IL7Ra', 'InsulinR', 'PDGFR', 'SARS_CoV2_RBD', 'EGFR', 'VirB8']

    my_targets = targets[run_args.my_task_id:len(targets):run_args.num_tasks]
    print("My targets: ", my_targets)

    for target in my_targets:
        df_target = df[df['target'].str.contains(target)]
        scaff_classes = list(set(df_target['scaff_class'].values))
        with open(f'/home/gridsan/fbirnbaum/joint-protein-embs/baker_results/{target}_prot_inter_sets.pickle', 'rb') as f:
            baker_prot_inter_sets = pickle.load(f)
        calc_dist_matrix(baker_prot_inter_sets, scaff_classes)
        # with mp.Pool(len(scaff_classes)) as pool:

        #     res_list = [
        #         pool.apply_async(calc_dist_matrix, (baker_prot_inter_sets, scaff_classes)) for _ in range(1)
        #     ]

        #     pool.close()
        #     pool.join()

        #     out_list = []
        #     for res in res_list:
        #         data = res.get()
        #         if data is not None:
        #             out_list.append(data)
        # print('Done: ', out_list)
                


