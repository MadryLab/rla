import src.models_and_optimizers as model_utils
from types import SimpleNamespace
from clip_main import get_wds_loaders
from transformers import EsmTokenizer
import src.data_utils as data_utils
import os
import torch
import sys
from tqdm import tqdm
import numpy as np
import json
from torch.cuda.amp import autocast
import json
import yaml
import pandas as pd
import esm as esmlib
import webdataset as wds
from terminator.data import noise as noise
from rla_utils import mask_peptide, mask_all, extract_knn, get_interaction_res, get_inter_dists, mask_peptide_struct, get_text_and_image_features, get_text_and_image_features_clip, calc_sim, calc_sim_clip, get_muts, score_mut, score_protein, compute_score, append_to_str, reverse_batch, reverse_batch_chain, segment_batch, test_batches
sys.path.insert(0, '/data1/groups/keatinglab/tools')
sys.path.insert(0, '/home/gridsan/fbirnbaum/TERMinator/scripts/design_sequence')
import argparse
import copy
import json
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train TERMinator!')
    parser.add_argument('--target', help='Target protein.', required=True)
    parser.add_argument('--dev', help='Device to run on', default='cuda:0', type=str)
    parser.add_argument('--data_root', help='Path to data dir', default='/data1/groups/keating_madry/wds/')
    parser.add_argument('--model_root', help='Path to model dir', default='/data1/groups/keating_madry/runs/new_blacklist', type=str)
    parser.add_argument('--esm_path', help='Path to model dir', default='/data1/groups/keating_madry/huggingface/esm2_t30_150M_UR50D', type=str)
    parser.add_argument('--config', help='COORDinator config', default='/home/gridsan/fbirnbaum/joint-protein-embs/terminator_configs/coordinator_broken_merge.json', type=str)
    parser.add_argument('--data_file', help='Raw binding data', default='/data1/groups/keating_madry/baker_designed_binders/all_data/retrospective_analysis/af2_rmsd_graphs1_data.sc')
    parser.add_argument('--seq_mask', help='Whether to mask peptide sequence', default=None)
    # parser.add_argument('--my_task_id', help='Task id for batch array', required=True, type=int)
    # parser.add_argument('--num_tasks', help='Number of jobs running simultaneously in batch array', required=True, type=int)
    run_args = parser.parse_args()
    # targets = ['IL7Ra', 'InsulinR', 'H3', 'EGFR', 'FGFR2', 'PDGFR']

    # print("my task id: ", args.my_task_id, " num tasks: ", args.num_tasks)
    # target = targets[args.my_task_id:len(targets):args.num_tasks]
    print("My target is: ", run_args.target)
    print("Dev: ", run_args.dev)
    print("Seq mask: ", run_args.seq_mask)
    target = run_args.target

    df = pd.read_csv(run_args.data_file, sep=' ')

    model_dir = "version_0/" 
    #model_dir = "6_1_pdb/version_0" 

    CLIP_MODE = False

    # need to edit these paths for supercloud  keating_madry_shared/models/9_17_big_coordinator
    ROOT = run_args.model_root

    root_path = os.path.join(ROOT, model_dir)
    path = os.path.join(root_path, "checkpoints/checkpoint_best.pt")
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
    args_dict['data_root'] = run_args.data_root
    args_dict['train_wds_path'] = f'baker_{target}_final.wds'
    args_dict['val_wds_path'] = f'baker_{target}_final.wds'
    args_dict['batch_size'] = 1
    args_dict['blacklist_file'] = ''
    for k in backwards_compat.keys():
        if k not in args_dict:
            args_dict[k] = backwards_compat[k]
    args = SimpleNamespace(**args_dict)

    coordinator_params = data_utils.get_coordinator_params(run_args.config)
    coordinator_params['num_positional_embeddings'] = args.gnn_num_pos_embs
    coordinator_params['zero_out_pos_embs']= args.gnn_zero_out_pos_embs
    coordinator_params['clip_mode'] = True
    

    args_dict['arch'] = run_args.esm_path
    trained_model = model_utils.load_model(path, args_dict['arch'], run_args.dev)
    tokenizer = EsmTokenizer.from_pretrained(args_dict['arch'])
    esm_arch = run_args.esm_path

    esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
    esm = esm.eval()
    if run_args.dev == 'cuda:0':
        esm = esm.cuda()

    if run_args.seq_mask is None:
        run_args.seq_mask = False
    seq_mask=run_args.seq_mask
    struct_mask=False
    top_k = 30
    focus = True
    remove_far = True
    threshold = 1
    weight_dists = False
    pep_weight = 1

    train_loader, val_loader, train_len, val_len = get_wds_loaders(args, coordinator_params, gpu=None, shuffle_train=False, val_only=True, return_count=False)
    for val_len, _ in enumerate(val_loader):
        continue
    val_len += 1
    val_len_c = copy.deepcopy(val_len)

    trained_model = trained_model.eval()

    lens = {}
    for i, b in tqdm(enumerate(val_loader), total=val_len):
        lens[b[0]['pdb_id'][0]] = len(b[0]['string_sequence'][0])
    max_len = max(lens.values())

    ## Calculate correlations between CLIP decoy scores and TM scores

    if CLIP_MODE:
        feature_getter = get_text_and_image_features_clip
    else:
        feature_getter = get_text_and_image_features
        
    torch.multiprocessing.set_sharing_strategy('file_system')
    all_scores = []
    models = []
    decoys = []
    # highlighted_keys = ["il2ra_site1_2b5i_sap_11_mot_HHH_b1_06312_000000036_0001_42_56_H_._HHH_b1_02527_0001_0000400021_0000001_0_0001",
    #                    "il2ra_site1_2b5i_sap_15_mot_HHH_b2_00070_000000071_0001_43_56_H_._HHH_b2_04717_0001_0000600030_0000001_0_0001_af2pred_0001_af2pred"]
    paired_scores = {}
    paired_res_scores = {}
    paired_res_scores_pep = {}
    paired_res_scores_prot = {}
    paired_res_scores_noise = {}
    paired_res_scores_pep_values = {}
    paired_res_scores_prot_values = {}
    plot_scores = []
    plot_weights = []
    plot_pep_mask = []
    plot_indices = []
    plot_X = []
    plot_seq = []
    MAX_LEN = max_len
    batch_iter = val_loader
    len_iter = val_len

    for i, batch in enumerate(tqdm(batch_iter, total=len_iter)):
        batch[1]['coords'][0][0] -= torch.mean(batch[1]['coords'][0][0], dim=[0,1])
        # if len(batch[0]['string_sequence'][0]) != 228:
        #     continue
        chain_lens = torch.zeros(batch[1]['coords'][0].shape[1]).to(device = batch[1]['coords'][0].device)
        chain_lens[batch[1]['seq_lens'][0][0]:] = 1
        chain_lens_mask = torch.ones(batch[1]['coords'][0].shape[1]).unsqueeze(0).to(dtype=torch.bool, device = batch[1]['coords'][0].device)    
        batch[1]['chain_lens'] = [chain_lens.unsqueeze(0), chain_lens_mask]
        
        pep_batch = copy.deepcopy(batch)
        prot_batch = copy.deepcopy(batch)
        pep_id = 0
        prot_id = 1
        pep_len = batch[1]['seq_lens'][0][pep_id]
        prot_len = batch[1]['seq_lens'][0][prot_id]
        pep_batch, prot_batch = segment_batch(pep_batch, prot_batch, pep_id, prot_id, pep_len, prot_len)
        # batch = append_to_str(batch)
        # pep_batch = append_to_str(pep_batch)
        # pep_batch = reverse_batch(pep_batch)
        t0 = time.time()
        pep_start = batch[1]['coords'][0][0][0][1]
        pep_end = batch[1]['coords'][0][0][pep_len-1][1]
        prot_coords = batch[1]['coords'][0][0][pep_len:][:,1]
        # if torch.min(torch.sum((prot_coords - pep_end)**2, -1)) > torch.min(torch.sum((prot_coords - pep_start)**2, -1)):
        #     pep_batch = reverse_batch(pep_batch)
        #     batch = reverse_batch_chain(batch, pep_len)
        noise_scores = []
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                # if batch[0]['pdb_id'][0] not in subset_list:
                #         continue
                output_dict = feature_getter(trained_model, tokenizer, batch, pdb=None, weight_dists=weight_dists, seq_mask=seq_mask, focus=focus, top_k=top_k, struct_mask=struct_mask, remove_far=remove_far, threshold=threshold, get_peptide_mask=True, dev=run_args.dev)
                # output_dict_all = feature_getter(trained_model, tokenizer, batch, pdb=None, weight_dists=False, focus=False, remove_far=False, threshold=threshold)
                output_dict_pep = feature_getter(trained_model, tokenizer, pep_batch, pdb=None, weight_dists=False, focus=False, remove_far=False, threshold=threshold, get_peptide_mask=True, dev=run_args.dev)
                output_dict_prot = feature_getter(trained_model, tokenizer, prot_batch, pdb=None, weight_dists=False, focus=False, remove_far=False, threshold=threshold, get_peptide_mask=True, dev=run_args.dev)
                # output_dict_all['text'] = output_dict_all['text'][:,5:,:]
                # output_dict_pep['text'] = output_dict_pep['text'][:,5:,:]
                # output_dict_all['seq_mask_no_burn_in'] = output_dict_all['seq_mask_no_burn_in'][:,5:]
                # output_dict_pep['seq_mask_no_burn_in'] = output_dict_pep['seq_mask_no_burn_in'][:,5:]
                score, scores, _, _, _, _, _, _ = compute_score(output_dict, batch, weight_dists, MAX_LEN, plot_scores=plot_scores, plot_weights=plot_weights, plot_pep_mask=plot_pep_mask, plot_indices=plot_indices, plot_X=plot_X, plot_seq=plot_seq, is_complex=True)
                # score, scores, _, _, _, _, _, _ = compute_score(output_dict_all, False, MAX_LEN, is_complex=False)
                pep_score, pep_scores, _, _, _, _, _, _ = compute_score(output_dict_pep, pep_batch, False, MAX_LEN, is_complex=False)
                prot_score, prot_scores, _, _, _, _, _, _ = compute_score(output_dict_prot, prot_batch, False, MAX_LEN, is_complex=False)

                pep_mask = output_dict['seq_mask_no_burn_in'][0,:pep_len]
                prot_mask = output_dict['seq_mask_no_burn_in'][0,pep_len:]
                # pep_mask = torch.ones((output_dict['weights'][:pep_len] > 0).shape).to(dtype=scores.dtype, device=scores.device)
                # prot_mask = torch.ones((output_dict['weights'][pep_len:] > 0).shape).to(dtype=scores.dtype, device=scores.device)
                pep_score_diff = torch.sum((scores[0, :pep_len] - pep_scores[0]) * pep_mask) / torch.sum(pep_mask)
                prot_score_diff = torch.sum((scores[0, pep_len:] - prot_scores[0]) * prot_mask) / torch.sum(prot_mask)
                # score = (pep_weight*pep_score_diff + prot_score_diff).cpu().item()
                # score = pep_score_diff.cpu().item()
                #pep_score_diffs = scores[:pep_len] pep_scores[]
    #             noise_scores.append(score)
    #             for i_noise in range(10):
                    
    #                 X_noise = noise.generate_noise('torsion_batch', 0.2, batch[0]['pdb_id'][0], i_noise, 5, output_dict['coord_data']['X'][0], batch[1]['coords'][1][0], chain_lens=batch[1]['seq_lens'][0])
    #                 output_dict['coord_data']['X'][0] += X_noise
    #                 output_dict['gnn'], _, _ = trained_model(None, output_dict['coord_data'])
    #                 output_dict['coord_data']['X'][0] -= X_noise
    #                 score, scores, _, _, _, _, _, _ = compute_score(output_dict, weight_dists, MAX_LEN, plot_scores=plot_scores, plot_weights=plot_weights, plot_pep_mask=plot_pep_mask, plot_indices=plot_indices, plot_X=plot_X, plot_seq=plot_seq, is_complex=True)
    #                 noise_scores.append(score)
                all_scores.append(score)  
                
                paired_scores[batch[0]['pdb_id'][0]] = score
                paired_res_scores[batch[0]['pdb_id'][0]] = [sc.cpu().item() for sc in scores[0]]
                paired_res_scores_noise[batch[0]['pdb_id'][0]] = noise_scores
                paired_res_scores_pep[batch[0]['pdb_id'][0]] = [sc.cpu().item() for sc in pep_scores[0]]
                paired_res_scores_prot[batch[0]['pdb_id'][0]] = [sc.cpu().item() for sc in prot_scores[0]]

                paired_res_scores_pep_values[batch[0]['pdb_id'][0]] = pep_score_diff.item()
                paired_res_scores_prot_values[batch[0]['pdb_id'][0]] = prot_score_diff.item()

                
    all_scores = np.array(all_scores)


    with open(f'baker_results/{target}_all_res_scores_k_30_new_blacklist_nopepseq_final.json', 'w') as f:
        json.dump(paired_res_scores, f)
        
    with open(f'baker_results/{target}_all_res_scores_k_30_new_blacklist_nopepseq_pep_final.json', 'w') as f:
        json.dump(paired_res_scores_pep, f)
        
    with open(f'baker_results/{target}_all_res_scores_k_30_new_blacklist_prot_nopepseq_final.json', 'w') as f:
        json.dump(paired_res_scores_prot, f)

    with open(f'baker_results/mlsb_{target}_all_scores_k_30_new_blacklist_nopepseq_final.json', 'w') as f:
        json.dump(paired_scores, f)

    with open(f'baker_results/mlsb_{target}_all_pep_scores_k_30_new_blacklist_nopepseq_final.json', 'w') as f:
        json.dump(paired_res_scores_pep_values, f)

    with open(f'baker_results/mlsb_{target}_all_prot_scores_k_30_new_blacklist_nopepseq_final.json', 'w') as f:
        json.dump(paired_res_scores_prot_values, f)