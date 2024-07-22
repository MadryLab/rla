import src.models_and_optimizers as model_utils
import yaml
from types import SimpleNamespace
from clip_main import get_wds_loaders
from transformers import EsmTokenizer
import src.data_utils as data_utils
import os
import torch
import sys
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
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import glob
import webdataset as wds

def mask_peptide(seq_batch, coords_batch, pdb):
    chain_len_dicts = {}
    chains, lens = torch.unique_consecutive(coords_batch['chain_lens'][0][0], return_counts=True)
    chain_len_dicts['protein'] = torch.max(lens)
    chain_len_dicts['peptide'] = torch.min(lens)
    peptide_len = chain_len_dicts['peptide']
    from_back = torch.argmin(lens) == 1
    if from_back:
        seq_batch['string_sequence'][0] = seq_batch['string_sequence'][0][:-peptide_len] + 'X'*peptide_len
        seq_batch['seq_loss_mask'][0][:,:-1*peptide_len] = False
        seq_batch['seq_loss_mask'][1][:,:-1*peptide_len] = False
    else:
        seq_batch['string_sequence'][0] = 'X'*peptide_len + seq_batch['string_sequence'][0][peptide_len:]
        seq_batch['seq_loss_mask'][0][:,peptide_len:] = False
        seq_batch['seq_loss_mask'][1][:,peptide_len:] = False
    return seq_batch

def mask_all(seq_batch, pdb):
    seq_batch['string_sequence'][0] = 'X'*len(seq_batch['string_sequence'][0])
    seq_batch['seq_loss_mask'][0][:,:] = False
    seq_batch['seq_loss_mask'][1][:,:] = False
    return seq_batch

# Extract kNN info
def extract_knn(X, eps, top_k):
    # Convolutional network on NCHW
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
    D = torch.sqrt(torch.sum(dX**2, 3) + eps)
    mask_2D = torch.ones(D.shape)
    D *= mask_2D

    # Identify k nearest neighbors (including self)
    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - mask_2D) * D_max
    D_neighbors, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)
    return D_neighbors, E_idx

def get_interaction_res(coords_batch, pdb, top_k, threshold=5, remove_far=False):
    chain_len_dicts = {}
    chains, lens = torch.unique_consecutive(coords_batch['chain_lens'][0][0], return_counts=True)
    chain_len_dicts['protein'] = torch.max(lens)
    chain_len_dicts['peptide'] = torch.min(lens)
    from_back = torch.argmin(lens) == 1
    top_k = min(top_k, coords_batch['coords'][0].size(1))
    D_neighbors, E_idx = extract_knn(coords_batch['coords'][0][:,:,1,:], eps=1e-6, top_k=top_k) 
    if from_back:
        interaction_res = set(range(chain_len_dicts['protein'], chain_len_dicts['protein']+chain_len_dicts['peptide']))
    else:
        interaction_res = set(range(0, chain_len_dicts['peptide']))
    # interaction_res = set(range(0, chain_len_dicts['protein']+chain_len_dicts['peptide']))
    if top_k == 0:
        return list(interaction_res)
    prot_to_add = set()
    for res in interaction_res:
        prot_to_add = prot_to_add.union(set(E_idx[0, res].tolist()))
    interaction_res = list(interaction_res.union(prot_to_add))
    if remove_far:
        to_remove = []
        for res in interaction_res:
            nother = 0
            opp = 1 - coords_batch['chain_lens'][0][0][res].item()
            for nres in E_idx[0, res]:
                if coords_batch['chain_lens'][0][0][nres].item() == opp:
                    nother += 1

            if nother < threshold:
                to_remove.append(res)
        for res in to_remove:
            interaction_res.remove(res)
    # Remove masked residues
    to_remove = []
    for res in interaction_res:
        if not coords_batch['coords'][1][0, res]:
            to_remove.append(res)
    for res in to_remove:
        interaction_res.remove(res)
    return interaction_res

def get_inter_dists(coords_batch, interaction_res, eps=1e-6):
    chain_len_dicts = {}
    chains, lens = torch.unique_consecutive(coords_batch['chain_lens'][0][0], return_counts=True)
    chain_len_dicts['protein'] = torch.max(lens)
    chain_len_dicts['peptide'] = torch.min(lens)
    from_back = torch.argmin(lens) == 1
    if from_back:
        pep_res = list(range(chain_len_dicts['protein'], chain_len_dicts['protein']+chain_len_dicts['peptide']))
        prot_res = ppe_res = list(range(0, chain_len_dicts['protein']))
    else:
        pep_res = list(range(0, chain_len_dicts['peptide']))
        prot_res = list(range(chain_len_dicts['peptide'], chain_len_dicts['peptide']+chain_len_dicts['protein']))
    pep_coords = coords_batch['coords'][0][:,pep_res,1,:]
    prot_coords = coords_batch['coords'][0][:,prot_res,1,:]
    inter_dists = []
    for res in range(chain_len_dicts['protein'] + chain_len_dicts['peptide']):
        if not res in interaction_res:
            inter_dists.append(np.nan)
            continue
        res_coord = coords_batch['coords'][0][:,res,1,:]
        if res in pep_res:
            dists = torch.sqrt(torch.sum((res_coord - prot_coords)**2, dim=2))
        else:
            dists = torch.sqrt(torch.sum((res_coord - pep_coords)**2, dim=2))
        inter_dists.append(torch.min(dists))
    inter_dists = np.array(inter_dists)
    # inter_dists = (np.nanmax(inter_dists) - inter_dists)
    # inter_dists = inter_dists / np.nanmax(inter_dists)
    inter_dists -= np.nanmin(inter_dists)
    inter_dists = 1 - (inter_dists / np.nanmax(inter_dists))
    # inter_dists -= np.nanmean(inter_dists)
    # inter_dists = np.abs(inter_dists) / np.nanstd(np.abs(inter_dists))
    inter_dists = np.nan_to_num(inter_dists, nan=0)
    return inter_dists
            

def mask_peptide_struct(coord_data, coords_batch, pdb, mask_prot=False):
    peptide_res = get_interaction_res(coords_batch, pdb, top_k=0)
    if not mask_prot:
        mask = torch.ones(coord_data['x_mask'].shape).to(dtype=coord_data['x_mask'].dtype, device=coord_data['x_mask'].device)
        mask[:,peptide_res] = 0
    else:
        mask = torch.zeros(coord_data['x_mask'].shape).to(dtype=coord_data['x_mask'].dtype, device=coord_data['x_mask'].device)
        mask[:,peptide_res] = 1
    coord_data['x_mask'] *= protein_mask
    coord_data['X'] *= protein_mask.unsqueeze(-1).unsqueeze(-1)
    return coord_data
    
def get_text_and_image_features(model, tokenizer, batch, pdb=None, seq_mask=None, struct_mask=None, focus=None, top_k=30, remove_far=False, threshold=5, weight_dists=False, get_peptide_mask=False, dev='cuda:0'):
    seq_batch, coords_batch = batch
    if seq_mask == 'peptide':
        seq_batch = mask_peptide(seq_batch, coords_batch, pdb)
    if seq_mask == 'all':
        seq_batch = mask_all(seq_batch, pdb)
    seqs = seq_batch['string_sequence']
    text_inp = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024+2)
    text_inp['position_ids'] = seq_batch['pos_embs'][0]
    text_inp = {k: v.to(dev) for k, v in text_inp.items()}
    coord_data = data_utils.construct_gnn_inp(coords_batch, device=dev, half_precision=True)
    if struct_mask=='peptide':
        coord_data = mask_peptide_struct(coord_data, coords_batch, pdb)
    gnn_features, text_features, logit_scale = model(text_inp, coord_data)
    new_text_features, _, new_text_mask = data_utils.postprocess_text_features(
        text_features=text_features, 
        inp_dict=text_inp, 
        tokenizer=tokenizer, 
        placeholder_mask=seq_batch['placeholder_mask'][0])
    if focus:
        interaction_res = get_interaction_res(coords_batch, pdb, top_k, remove_far = remove_far, threshold = threshold)
        interaction_mask = torch.zeros(new_text_mask.shape).to(dtype=new_text_mask.dtype, device=new_text_mask.device)
        interaction_mask[:,interaction_res] = 1
        new_text_mask *= interaction_mask
    if weight_dists:
        weights = torch.from_numpy(get_inter_dists(coords_batch, interaction_res)).to(device=gnn_features.device)
    else:
        weights = None
    if get_peptide_mask:
        peptide_mask_ind = get_interaction_res(coords_batch, pdb, 0)
        peptide_mask = torch.zeros_like(new_text_mask.bool())
        peptide_mask[0][peptide_mask_ind] = True
    else:
        peptide_mask = None
    return {
        'text': new_text_features, # text feature
        'gnn': gnn_features, # gnn feature
        'seq_mask_with_burn_in': seq_batch['seq_loss_mask'][0], # sequence mask of what's supervised
        'coord_mask_with_burn_in': coords_batch['coords_loss_mask'][0], # coord mask of what's supervised
        'seq_mask_no_burn_in': new_text_mask.bool(), # sequence mask of what's valid (e.g., not padded)
        'coord_mask_no_burn_in': coords_batch['coords'][1], # coord mask of what's valid
        'weights': weights, # distance weights, if requested
        'peptide_mask': peptide_mask,
        'coord_data': coord_data
    }

def get_text_and_image_features_clip(model, tokenizer, batch, dev='cuda:0'):
    seq_batch, coords_batch = batch
    seqs = seq_batch['string_sequence']
    text_inp = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024+2)
    text_inp['position_ids'] = seq_batch['pos_embs'][0]
    text_inp = {k: v.to(dev) for k, v in text_inp.items()}
    coord_data = data_utils.construct_gnn_inp(coords_batch, device=dev, half_precision=True)
    gnn_features, text_features, logit_scale = model(text_inp, coord_data)
    return {
        'text': text_features, # text feature
        'gnn': gnn_features, # gnn feature
        'seq_mask_with_burn_in': seq_batch['seq_loss_mask'][0], # sequence mask of what's supervised
        'coord_mask_with_burn_in': coords_batch['coords_loss_mask'][0], # coord mask of what's supervised
    }

cos = torch.nn.CosineSimilarity()
def calc_sim(all_outputs):
    all_sims = []
    all_sims_burn = []
    for output in all_outputs:
        t = output['text'][output['seq_mask_no_burn_in']]
        g = output['gnn'][output['coord_mask_no_burn_in']]
        sim = (t.unsqueeze(1) @ g.unsqueeze(-1)).squeeze(1).squeeze(1)
        all_sims.append(torch.mean(sim))

    return all_sims

def calc_sim_clip(all_outputs):
    all_sims = []
    all_sims_burn = []
    for output in all_outputs:
        t = output['text']
        g = output['gnn']
        print(t.shape, g.shape)
        ### ??? ###
        sim = (t.unsqueeze(1) @ g.unsqueeze(-1)).squeeze(1).squeeze(1)
        all_sims.append(torch.mean(sim))
        ### ??? ###
    return all_sims
        
## ESM mutation analysis functions
def get_muts(wt, mut):
    inds = []
    muts = []
    for i, (wchar, mchar) in enumerate(zip(wt, mut)):
        if wchar != mchar:
            inds.append(i)
            muts.append(mchar)
    return inds, muts

def score_mut(wt, idx, mt, token_probs, alphabet):
    wt = wt[idx]
    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # add 1 for BOS
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    return score.item()

def score_protein(idxs, mts, wt_seq, model, alphabet):

    # inference for each model
    batch_converter = alphabet.get_batch_converter()

    data = [
        ("protein1", wt_seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    all_token_probs = []
    for i in tqdm(range(batch_tokens.size(1))):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(
                model(batch_tokens_masked.cuda())["logits"], dim=-1
            )
        all_token_probs.append(token_probs[:, i])  # vocab size
    token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
    esm_predictions = []
    for idx_list, mut_list in zip(idxs, mts):
        mut_score = 0
        for idx, mut in zip(idx_list, mut_list):
            mut_score += score_mut(wt_seq, idx, mut, token_probs, alphabet)
        if len(idx_list) == 0:
            mut_score = np.nan
        esm_predictions.append(mut_score)
    return esm_predictions

def compute_score(output_dict, batch, weight_dists, MAX_LEN, pep_weight=1, plot_scores=None, plot_weights=None, plot_pep_mask=None, plot_indices=None, plot_X=None, plot_seq=None, is_complex=False):
    text_feat = output_dict['text']
    gnn_feat =  output_dict['gnn'][:, :text_feat.shape[1]] # remove tail padding
    scores = (text_feat.unsqueeze(2) @ gnn_feat.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    if is_complex:
        pep_scores = list(scores[output_dict['peptide_mask']].cpu().numpy())
        prot_scores = list(scores[~output_dict['peptide_mask']].cpu().numpy())
        pep_len = len(pep_scores)
        inter_len = (MAX_LEN - pep_len - len(prot_scores))
        plot_pep_mask += list(output_dict['peptide_mask'][0][:pep_len].cpu().numpy()) + inter_len*[True] + list(output_dict['peptide_mask'][0][pep_len:].cpu().numpy())
        pep_scores += inter_len*[-1]
        plot_scores += (pep_scores + prot_scores)
        plot_indices += list(range(MAX_LEN)) #output_dict['peptide_mask'].shape[1]
    
    X = batch[1]['coords'][0][0].cpu().numpy()
    if is_complex and inter_len > 0:
        plot_X.append(np.concatenate((X[:pep_len], -1*np.ones((inter_len, X.shape[1], X.shape[2])), X[pep_len:]), axis=0))
    elif is_complex and inter_len == 0:
        plot_X.append(X)
    if is_complex and plot_X[-1].shape[0] != MAX_LEN:
        raise ValueError
    if is_complex:
        plot_seq.append(batch[0]['string_sequence'][0][:pep_len] + 'X'*inter_len + batch[0]['string_sequence'][0][pep_len+25:])
    if weight_dists:
        # weights = output_dict['weights'] / torch.sum( output_dict['weights'])
        if is_complex:
            plot_weights += list(output_dict['weights'][:pep_len].cpu().numpy()) + (MAX_LEN - pep_len - len(prot_scores))*[-1] + list(output_dict['weights'][pep_len:].cpu().numpy())
        if len(plot_scores) != len(plot_weights) or len(plot_indices) != len(plot_weights):
            print(len(plot_scores), len(plot_weights), len(plot_indices))
            raise ValueError
        # output_dict['weights'][:] = 0
        # output_dict['weights'][10] = 1
        # cur_weights = torch.from_numpy(np.concatenate([var_weights[:pep_len], var_weights[len(pep_scores):]])).to(output_dict['weights'].device)
        cur_weights = output_dict['weights']
        # scores *= cur_weights
        if is_complex:
            cur_pep_weights = cur_weights * output_dict['peptide_mask'][0]
            cur_pep_weights = cur_pep_weights / torch.sum(cur_pep_weights)
            cur_prot_weights = cur_weights * ~output_dict['peptide_mask'][0]
            cur_prot_weights = cur_prot_weights / torch.sum(cur_prot_weights)
            cur_weights = cur_pep_weights + cur_prot_weights
        scores *= cur_weights
        # scores += scores*cur_weights # output_dict['weights']
        # scores /= 2

    if is_complex:
        pep_scores = scores * output_dict['peptide_mask']
        pep_seq_mask = output_dict['seq_mask_no_burn_in'].float() * output_dict['peptide_mask']
        prot_scores = scores * ~output_dict['peptide_mask']
        prot_seq_mask = output_dict['seq_mask_no_burn_in'].float() * ~output_dict['peptide_mask']
        # special_pep_mask = torch.ones_like(output_dict['seq_mask_no_burn_in']).to(dtype=torch.float32)
        # special_pep_mask[0,:18] = 1
        # if design_set == design_sets[-1]:
        #     pep_scores *= special_pep_mask
        scores = pep_scores + prot_scores
        pep_score = pep_weight*(pep_scores * pep_seq_mask).sum(1)/pep_seq_mask.sum(1)
        prot_score = (prot_scores * prot_seq_mask).sum(1)/prot_seq_mask.sum(1)
        if not torch.isnan(prot_score).item():
            score = (pep_score.cpu().item() + prot_score.cpu().item()) / 2
        else:
            score = pep_score.cpu().item()
    else:
        score = (scores * output_dict['seq_mask_no_burn_in'].float()).sum(1)/output_dict['seq_mask_no_burn_in'].sum(1)
        score = score.cpu().item()
    return score, scores, plot_scores, plot_weights, plot_pep_mask, plot_indices, plot_X, plot_seq

def append_to_str(batch):
    # batch = copy.deepcopy(batch)
    batch[0]['string_sequence'] = [5*'G' + batch[0]['string_sequence'][0]]
    batch_pad = torch.zeros(1, 5).to(dtype=batch[0]['pos_embs'][0].dtype, device=batch[0]['pos_embs'][0].device)
    batch_pad_mask = torch.ones(1, 5).to(dtype=batch[0]['pos_embs'][1].dtype, device=batch[0]['pos_embs'][1].device)
    batch[0]['pos_embs'] = [torch.cat([-1 + batch_pad, batch[0]['pos_embs'][0]], 1).to(dtype=batch[0]['pos_embs'][0].dtype),
                                torch.cat([batch_pad_mask, batch[0]['pos_embs'][1]], 1).to(dtype=batch[0]['pos_embs'][1].dtype)]
    batch[0]['placeholder_mask'] = [torch.cat([batch_pad_mask, batch[0]['placeholder_mask'][0]], 1).to(dtype=batch[0]['placeholder_mask'][0].dtype), 
                                        torch.cat([batch_pad_mask, batch[0]['placeholder_mask'][1]], 1).to(dtype=batch[0]['placeholder_mask'][1].dtype)]
    batch[0]['seq_loss_mask'] = [torch.cat([batch_pad_mask, batch[0]['seq_loss_mask'][0]], 1).to(dtype=batch[0]['seq_loss_mask'][0].dtype),
                                     torch.cat([batch_pad_mask, batch[0]['seq_loss_mask'][1]], 1).to(dtype=batch[0]['seq_loss_mask'][1].dtype)]
    batch[0]['seq_to_coords'] = [torch.cat([-1 + batch_pad, batch[0]['seq_to_coords'][0]], 1).to(dtype=batch[0]['seq_to_coords'][0].dtype),
                                     torch.cat([batch_pad_mask, batch[0]['seq_to_coords'][1]], 1).to(dtype=batch[0]['seq_to_coords'][0].dtype)]
    return batch

def reverse_batch(batch):
    batch = copy.deepcopy(batch)
    batch[0]['string_sequence'] = [batch[0]['string_sequence'][0][::-1]]
    # batch[0]['pos_embs'] = [torch.flip(batch[0]['pos_embs'][0], 1), torch.flip(batch[0]['pos_embs'][1], 1)]
    # batch[0]['placeholder_mask'] = [torch.flip(batch[0]['placeholder_mask'][0], 1), torch.flip(batch[0]['placeholder_mask'][1], 1)]
    # batch[0]['seq_loss_mask'] = [torch.flip(batch[0]['seq_loss_mask'][0], 1), torch.flip(batch[0]['seq_loss_mask'][1], 1)]
    # batch[0]['seq_to_coords'] = [torch.flip(batch[0]['seq_to_coords'][0], 1), torch.flip(batch[0]['seq_to_coords'][1], 1)]
    batch[1]['coords'] = [torch.flip(batch[1]['coords'][0], [1]).to(dtype=batch[1]['coords'][0].dtype, device=batch[1]['coords'][0].device), torch.flip(batch[1]['coords'][1], [1]).to(dtype=batch[1]['coords'][1].dtype, device=batch[1]['coords'][1].device)]
    # batch[1]['res_info'] = [batch[1]['res_info'][0][::-1]]
    # batch[1]['seq_lens'] = [batch[1]['seq_lens'][0][::-1]]
    # batch[1]['coords_loss_mask'] = [torch.flip(batch[1]['coords_loss_mask'][0], 1), torch.flip(batch[1]['coords_loss_mask'][1], 1)]
    # batch[1]['coords_to_seq'] = [torch.flip(batch[1]['coords_to_seq'][0], 1), torch.flip(batch[1]['coords_to_seq'][1], 1)]
    # batch[1]['chain_dict']['begin'] = [batch[1]['chain_dict']['begin'][0].narrow(1, 0, pep_len), batch[1]['chain_dict']['begin'][1].narrow(1, 0, pep_len)]
    # batch[1]['chain_dict']['end'] = [batch[1]['chain_dict']['end'][0].narrow(1, 0, pep_len), batch[1]['chain_dict']['end'][1].narrow(1, 0, pep_len)]
    # batch[1]['chain_dict']['end'][0][0,-1] = batch[1]['chain_dict']['end'][0][0,-2] 
    # batch[1]['chain_dict']['singles'] = [batch[1]['chain_dict']['singles'][0].narrow(1, 0, pep_len), batch[1]['chain_dict']['singles'][1].narrow(1, 0, pep_len)]
    # batch[1]['chain_dict']['ids'] = [batch[1]['chain_dict']['ids'][0].narrow(1, 0, pep_len) - pep_id, batch[1]['chain_dict']['ids'][1].narrow(1, 0, pep_len)]
    # batch[1]['chain_lens'] = [batch[1]['chain_lens'][0].narrow(1, 0, pep_len) - pep_id, batch[1]['chain_lens'][1].narrow(1, 0, pep_len)]
    return batch
    
def reverse_batch_chain(batch, pep_len):
    batch = copy.deepcopy(batch)
    batch[0]['string_sequence'] = [batch[0]['string_sequence'][0][:pep_len][::-1] + batch[0]['string_sequence'][0][pep_len:]]
    batch[1]['coords'] = [torch.cat([torch.flip(batch[1]['coords'][0][:,:pep_len], [1]), batch[1]['coords'][0][:,pep_len:]], 1).to(dtype=batch[1]['coords'][0].dtype, device=batch[1]['coords'][0].device),
                          torch.cat([torch.flip(batch[1]['coords'][1][:,:pep_len], [1]), batch[1]['coords'][1][:,pep_len:]], 1).to(dtype=batch[1]['coords'][1].dtype, device=batch[1]['coords'][1].device)]
    return batch

def segment_batch(pep_batch, prot_batch, pep_id, prot_id, pep_len, prot_len):
    pep_batch[0]['string_sequence'] = [pep_batch[0]['string_sequence'][0][:pep_len]]
    pep_batch[0]['pos_embs'] = [pep_batch[0]['pos_embs'][0].narrow(1, 0, pep_len+2), pep_batch[0]['pos_embs'][1].narrow(1, 0, pep_len+2)]
    pep_batch[0]['placeholder_mask'] = [pep_batch[0]['placeholder_mask'][0].narrow(1, 0, pep_len), pep_batch[0]['placeholder_mask'][1].narrow(1, 0, pep_len)]
    pep_batch[0]['seq_loss_mask'] = [pep_batch[0]['seq_loss_mask'][0].narrow(1, 0, pep_len), pep_batch[0]['seq_loss_mask'][1].narrow(1, 0, pep_len)]
    pep_batch[0]['seq_to_coords'] = [pep_batch[0]['seq_to_coords'][0].narrow(1, 0, pep_len), pep_batch[0]['seq_to_coords'][1].narrow(1, 0, pep_len)]
    pep_batch[1]['coords'] = [pep_batch[1]['coords'][0].narrow(1, 0, pep_len), pep_batch[1]['coords'][1].narrow(1, 0, pep_len)]
    pep_batch[1]['res_info'] = [pep_batch[1]['res_info'][0][:pep_len]]
    pep_batch[1]['seq_lens'] = [[pep_batch[1]['seq_lens'][0][pep_id]]]
    pep_batch[1]['coords_loss_mask'] = [pep_batch[1]['coords_loss_mask'][0].narrow(1, 0, pep_len), pep_batch[1]['coords_loss_mask'][1].narrow(1, 0, pep_len)]
    pep_batch[1]['coords_to_seq'] = [pep_batch[1]['coords_to_seq'][0].narrow(1, 0, pep_len), pep_batch[1]['coords_to_seq'][1].narrow(1, 0, pep_len)]
    pep_batch[1]['chain_dict']['begin'] = [pep_batch[1]['chain_dict']['begin'][0].narrow(1, 0, pep_len), pep_batch[1]['chain_dict']['begin'][1].narrow(1, 0, pep_len)]
    pep_batch[1]['chain_dict']['end'] = [pep_batch[1]['chain_dict']['end'][0].narrow(1, 0, pep_len), pep_batch[1]['chain_dict']['end'][1].narrow(1, 0, pep_len)]
    pep_batch[1]['chain_dict']['end'][0][0,-1] = pep_batch[1]['chain_dict']['end'][0][0,-2] 
    pep_batch[1]['chain_dict']['singles'] = [pep_batch[1]['chain_dict']['singles'][0].narrow(1, 0, pep_len), pep_batch[1]['chain_dict']['singles'][1].narrow(1, 0, pep_len)]
    pep_batch[1]['chain_dict']['ids'] = [pep_batch[1]['chain_dict']['ids'][0].narrow(1, 0, pep_len) - pep_id, pep_batch[1]['chain_dict']['ids'][1].narrow(1, 0, pep_len)]
    pep_batch[1]['chain_lens'] = [pep_batch[1]['chain_lens'][0].narrow(1, 0, pep_len) - pep_id, pep_batch[1]['chain_lens'][1].narrow(1, 0, pep_len)]
    
    prot_batch[0]['string_sequence'] = [prot_batch[0]['string_sequence'][0][pep_len+25:]]
    prot_batch_cut_pe = prot_batch[0]['pos_embs'][0][0][pep_len+26] - 1
    prot_batch_cut_stc = prot_batch[0]['seq_to_coords'][0].narrow(1, pep_len, prot_len)[0][0]
    prot_batch[0]['pos_embs'] = [prot_batch[0]['pos_embs'][0].narrow(1, pep_len+25, prot_len+2) - prot_batch_cut_pe, prot_batch[0]['pos_embs'][1].narrow(1, pep_len+25, prot_len+2)]
    prot_batch[0]['pos_embs'][0][0][0] = 0
    prot_batch[0]['placeholder_mask'] = [prot_batch[0]['placeholder_mask'][0].narrow(1, pep_len+25, prot_len), prot_batch[0]['placeholder_mask'][1].narrow(1, pep_len+25, prot_len)]
    prot_batch[0]['seq_loss_mask'] = [prot_batch[0]['seq_loss_mask'][0].narrow(1, pep_len, prot_len), prot_batch[0]['seq_loss_mask'][1].narrow(1, pep_len, prot_len)]
    prot_batch[0]['seq_to_coords'] = [prot_batch[0]['seq_to_coords'][0].narrow(1, pep_len, prot_len) - prot_batch_cut_stc, prot_batch[0]['seq_to_coords'][1].narrow(1, pep_len, prot_len)]
    prot_batch[1]['coords'] = [prot_batch[1]['coords'][0].narrow(1, pep_len, prot_len), prot_batch[1]['coords'][1].narrow(1, pep_len, prot_len)]
    prot_batch[1]['res_info'] = [prot_batch[1]['res_info'][0][pep_len:]]
    prot_batch[1]['seq_lens'] = [[prot_batch[1]['seq_lens'][0][prot_id]]]
    prot_batch[1]['coords_loss_mask'] = [prot_batch[1]['coords_loss_mask'][0].narrow(1, pep_len, prot_len), prot_batch[1]['coords_loss_mask'][1].narrow(1, pep_len, prot_len)]
    prot_batch[1]['coords_to_seq'] = [prot_batch[1]['coords_to_seq'][0].narrow(1, pep_len, prot_len) - prot_batch_cut_stc, prot_batch[1]['coords_to_seq'][1].narrow(1, pep_len, prot_len)]
    prot_batch[1]['chain_dict']['begin'] = [prot_batch[1]['chain_dict']['begin'][0].narrow(1, pep_len, prot_len) - prot_batch_cut_stc, prot_batch[1]['chain_dict']['begin'][1].narrow(1, pep_len, prot_len)]
    prot_batch[1]['chain_dict']['end'] = [prot_batch[1]['chain_dict']['end'][0].narrow(1, pep_len, prot_len) - prot_batch_cut_stc, prot_batch[1]['chain_dict']['end'][1].narrow(1, pep_len, prot_len)]
    prot_batch[1]['chain_dict']['singles'] = [prot_batch[1]['chain_dict']['singles'][0].narrow(1, pep_len, prot_len), prot_batch[1]['chain_dict']['singles'][1].narrow(1, pep_len, prot_len)]
    prot_batch[1]['chain_dict']['ids'] = [prot_batch[1]['chain_dict']['ids'][0].narrow(1, pep_len, prot_len) - prot_id, prot_batch[1]['chain_dict']['ids'][1].narrow(1, pep_len, prot_len)]
    prot_batch[1]['chain_lens'] = [prot_batch[1]['chain_lens'][0].narrow(1, pep_len, prot_len) - prot_id, prot_batch[1]['chain_lens'][1].narrow(1, pep_len, prot_len)]

    return pep_batch, prot_batch

def test_batches(pep_batch, prot_batch, pep_batch_r, prot_batch_r):
    k0 = pep_batch[0].keys()
    k1 = pep_batch[1].keys()
    for name, test, real in zip(['pep', 'prot'], [pep_batch, prot_batch], [pep_batch_r, prot_batch_r]):
        for i in [0,1]:
            if i == 0:
                for k in k0:
                    if k == 'string_sequence' or k == 'pdb_id':
                        if real[i][k] != test[i][k]:
                            print(name, i, k)
                            raise ValueError
                    else:
                        if (not ((real[i][k][0] == test[i][k][0]).all())) or (not ((real[i][k][1] == test[i][k][1]).all())):
                            print(name, i, k)
                            raise ValueError
            else:
                for k in k1:
                    if k == 'chain_dict':
                        for kk in test[i][k].keys():
                            if (not ((real[i][k][kk][0] == test[i][k][kk][0]).all())) or (not ((real[i][k][kk][1] == test[i][k][kk][1]).all())):
                                print(name, i, k)
                                raise ValueError
                    elif k == 'res_info' or k == 'seq_lens':
                        if real[i][k] != test[i][k]:
                            print(name, i, k)
                            raise ValueError
                    else:
                        if (not ((real[i][k][0] == test[i][k][0]).all())) or (not ((real[i][k][1] == test[i][k][1]).all())):
                            print(name, i, k)
                            raise ValueError
        print(f'made it: {name}!')
