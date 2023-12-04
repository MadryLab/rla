import sys
import torch
import h5py
import numpy as np
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, Sampler
from terminator.utils.model.default_hparams import DEFAULT_MODEL_HPARAMS, DEFAULT_TRAIN_HPARAMS
from terminator.data.data import _ingraham_featurize
from terminator.models.layers.utils import extract_knn, extract_idxs, per_node_to_all_comb
import json
from torch.nn.utils.rnn import pad_sequence

def construct_gnn_inp(coords_output, device, half_precision):
    X, X_mask = coords_output['coords']
    X = X.to(device)
    if half_precision:
        X = X.half()
    chain_idx = coords_output['seq_lens']

    chain_dict = {k: v[0].to(device).long() for k,v in coords_output['chain_dict'].items()}
    chain_dict['singles'] = chain_dict['singles'].float()
    return {
        'X': X,
        'x_mask': X_mask.to(device).float(),
        'chain_idx': chain_idx,
        'chain_dict': chain_dict,
    }


def load_params(params_file, default_hparams):
    with open(params_file) as fp:
        hparams = json.load(fp)
    for key, default_val in default_hparams.items():
        if key not in hparams:
            hparams[key] = default_val
    return hparams
    
def get_coordinator_params(params_file):
    return load_params(params_file, DEFAULT_MODEL_HPARAMS)

def postprocess_text_features(text_features, inp_dict, tokenizer, placeholder_mask, min_length=None):
    # remove class token, eos token
    text_features, input_ids, text_mask = _adjust_text_features(text_features, inp_dict, tokenizer)
    # remove placeholders
    text_features, input_ids, text_mask = _remove_placeholders(text_features, input_ids, text_mask, placeholder_mask, min_length=min_length)
    return text_features, input_ids, text_mask

def _adjust_text_features(text_features, inp_dict, tokenizer):
    mask = inp_dict['attention_mask'].clone()
    toks = inp_dict['input_ids']
    eos_token = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    mask[toks == eos_token] = 0
    # ignore class token and last eos token
    mask = mask[:, 1:-1]
    text_features = text_features[:, 1:-1]
    toks = toks[:, 1:-1]
    return text_features, toks, mask

def _remove_placeholders(text_features, input_ids, text_mask, placeholder_mask, min_length):
    B = placeholder_mask.shape[0]
    filtered, new_masks, tokens = [], [], []
    for b in range(B):
        p_m = placeholder_mask[b][:len(text_features[b])] # placeholder mask sometimes has tail padding
        #new_text_feat = text_features[b][p_m]
        filtered.append(text_features[b][p_m])
        tokens.append(input_ids[b][p_m])
        new_masks.append(text_mask[b][p_m])
    filtered = pad_seq_with_len(filtered, min_length=min_length)
    new_masks = pad_seq_with_len(new_masks, min_length=min_length)
    tokens = pad_seq_with_len(tokens, min_length=min_length)
    return filtered, tokens, new_masks

def pad_seq_with_len(arr, min_length=None):
    arr = pad_sequence(arr, batch_first=True)
    if min_length is not None:
        if arr.shape[1] < min_length:
            pads = [0 for _ in range(2*len(arr.shape))]
            pads[3] = min_length - arr.shape[1]
            arr = torch.nn.functional.pad(arr, pads)
    return arr

def get_text_and_image_features(batch, tokenizer, model, hparams):
    with torch.no_grad():
        with autocast(enabled=True, dtype=torch.float16):
            batch = {
                'coords': batch[0],
                'coords_mask': batch[1],
                'chain_len': batch[2],
                'seq_len': batch[3],
                'seq': [u['seq'] for u in batch[4]],
                'inds_reduce': batch[5],
                'inds_expand': batch[6],
                'inds_transpose': batch[7],
                'inds_duplicate': batch[8],
                'inds_single': batch[9],
                'mask_reduced': batch[10]
            }

            seqs = batch['seq']
            text_inp = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=hparams['training']['max_seq_len']+2)
            text_inp = {k: v.to('cuda') for k, v in text_inp.items()}
            coord_data, max_seq_len = construct_gnn_inp(batch, device='cuda', half_precision=True)
            image_features, text_features, logit_scale = model(text_inp, coord_data, max_seq_len=max_seq_len)
            new_text_features, new_text_mask = adjust_text_features(text_features, text_inp, tokenizer)
    return new_text_features, image_features, coord_data['x_mask']

def get_potts_model(model, coord_data, max_seq_len, init_node_embeddings=None):
    coordinator = model.get_GNN()
    with torch.no_grad():
        with autocast(enabled=True, dtype=torch.float16):
            etabs, E_idx, _ = coordinator(coord_data, max_seq_len=max_seq_len, init_node_embeddings=init_node_embeddings)
    return etabs, E_idx