import src.models_and_optimizers as model_utils
import src.data_utils as data_utils
from types import SimpleNamespace
from clip_main import get_wds_loaders
from transformers import EsmTokenizer
import src.data_utils as data_utils
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sys
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast
import pandas as pd
import esm as esmlib
import webdataset as wds
from terminator.data import noise as noise
from rla_utils import get_text_and_image_features
sys.path.insert(0, '/data1/groups/keatinglab/tools')
from run_dockq import run_dockq
sys.path.insert(0, '/home/gridsan/fbirnbaum/TERMinator/scripts/design_sequence')
import argparse
import copy
import math

from transformers import AutoTokenizer, EsmForMaskedLM
import esm as esmlib

import train_converter
ConverterModel = train_converter.ConverterModel
ConverterTransformerModel = train_converter.ConverterTransformerModel
ConverterTransformerEncoderModel = train_converter.ConverterTransformerEncoderModel
EsmLMHead = train_converter.EsmLMHead
ReshapeBaseHead = train_converter.ReshapeBaseHead


def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def run_iter(batch, trained_model, tokenizer, converter, esm, lm_head_base, lm_head_converter, lm_head_new, base_optimizer, converter_optimizer, new_optimizer, grad=True):
    
    chain_lens = torch.zeros(batch[1]['coords'][0].shape[1]).to(device = batch[1]['coords'][0].device)
    chain_lens[batch[1]['seq_lens'][0][0]:] = 1
    chain_lens_mask = torch.ones(batch[1]['coords'][0].shape[1]).unsqueeze(0).to(dtype=torch.bool, device = batch[1]['coords'][0].device)    
    batch[1]['chain_lens'] = [chain_lens.unsqueeze(0), chain_lens_mask]
    criterion = torch.nn.NLLLoss(reduction='none')
    with torch.no_grad():
        with autocast(dtype=torch.float16):
            output_dict = get_text_and_image_features(trained_model, tokenizer, batch, pdb=None, weight_dists=False, seq_mask=None, focus=False, top_k=30, struct_mask=None, 
                                                          remove_far=False, threshold=1, dev=dev)
        seq_embeddings = output_dict['text'].squeeze(0)
        seqs = batch[0]['string_sequence']
        text_inp = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024+2)
        text_inp['position_ids'] = batch[0]['pos_embs'][0]
        text_inp = {k: v.to(dev) for k, v in text_inp.items()}
        base_seq_in = trained_model.get_text_features_no_proj(text_inp)
        base_seq_in, _, _ = data_utils.postprocess_text_features(text_features=base_seq_in, inp_dict=text_inp, tokenizer=tokenizer, placeholder_mask=batch[0]['placeholder_mask'][0])
        base_seq_in = base_seq_in.squeeze(0)

        struct_embeddings = output_dict['gnn'][output_dict['coord_mask_with_burn_in']]
        seq_embeddings = output_dict['text'].squeeze(0)

        ind = batch[0]['placeholder_mask'][0][0].cpu().numpy()
        esm_seq = "".join(np.array(list(batch[0]['string_sequence'][0]))[ind])
        esm_data = [('prot', esm_seq)]
        _, _, batch_tokens = batch_converter(esm_data)
        batch_tokens = batch_tokens.to('cuda:0')
        esm_results = esm(batch_tokens, repr_layers=[30], return_contacts=False)
        esm_seq_in = esm_results['representations'][30].squeeze(0)[1:-1]
        # esm_seq_in = esm_seq_in[batch[0]['placeholder_mask'][0][0]]
        target = batch_tokens[0,1:-1]
        converter_seq_in = converter(struct_embeddings.unsqueeze(0), seq_embeddings.unsqueeze(0)).squeeze(0)
        # new_seq_in = copy.deepcopy(converter_seq_in)

    try:
        base_pred = torch.softmax(lm_head_base(base_seq_in), dim=-1)
        converter_pred = torch.softmax(lm_head_converter(converter_seq_in), dim=-1)
        # new_pred = torch.softmax(lm_head_new(esm_seq_in), dim=-1)
        base_loss = criterion(base_pred, target).mean()
        converter_loss = criterion(converter_pred, target).mean()
        # new_loss = criterion(new_pred, target).mean()
    except Exception as E:
        print(E)
        print(batch[0]['pdb_id'][0])
        print(len(batch[0]['string_sequence'][0]))

    if grad:
        base_optimizer.zero_grad()
        base_loss.backward()
        base_optimizer.step()
        converter_optimizer.zero_grad()
        converter_loss.backward()
        converter_optimizer.step()
        # new_optimizer.zero_grad()
        # new_loss.backward()
        # new_optimizer.step()
        
    base_pred_tokens = torch.argmax(base_pred, dim=-1)
    base_nsr = (base_pred_tokens == target).sum() / target.numel()
    converter_pred_tokens = torch.argmax(converter_pred, dim=-1)
    converter_nsr = (converter_pred_tokens == target).sum() / target.numel()
    # new_pred_tokens = torch.argmax(new_pred, dim=-1)
    # new_nsr = (new_pred_tokens == target).sum() / target.numel()
    new_nsr = torch.Tensor([0])
    new_loss = torch.Tensor([0])
    
    return base_loss.item(), base_nsr.item(), converter_loss.item(), converter_nsr.item(), new_loss.item(), new_nsr.item()
    

if __name__ == '__main__':

    esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm = esm.cuda()

    run_dir = '/home/gridsan/fbirnbaum/joint-protein-embs/converter_runs/multichain_transformer_lr_3_reg_4_drop_2_mse_noproj_esm'
    val_wds = 'multichain_clip_val.wds'
    train_wds = 'multichain_clip_train.wds'
    data_root = '/data1/groups/keating_madry/wds/'
    rla_root = '/data1/groups/keating_madry/runs/new_blacklist'
    esm_path = '/data1/groups/keating_madry/huggingface/esm2_t30_150M_UR50D'
    config = '/home/gridsan/fbirnbaum/joint-protein-embs/terminator_configs/coordinator_broken_merge.json'
    model_type = 'encoder'
    text_feat_type = 'no_proj'
    dev = 'cuda:0'

    model_dir = "version_0/" 
    CLIP_MODE = False
    ROOT = rla_root
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
    args_dict['data_root'] = data_root
    args_dict['train_wds_path'] = train_wds
    args_dict['val_wds_path'] = val_wds
    args_dict['batch_size'] = 1
    args_dict['blacklist_file'] = ''
    for k in backwards_compat.keys():
        if k not in args_dict:
            args_dict[k] = backwards_compat[k]
    rla_args = SimpleNamespace(**args_dict)
    rla_args.coordinator_hparams = config

    coordinator_params = data_utils.get_coordinator_params(rla_args.coordinator_hparams)
    coordinator_params['num_positional_embeddings'] = rla_args.gnn_num_pos_embs
    coordinator_params['zero_out_pos_embs']= rla_args.gnn_zero_out_pos_embs
    coordinator_params['clip_mode'] = True


    args_dict['arch']= esm_path
    trained_model = model_utils.load_model(path, args_dict['arch'], dev)
    tokenizer = EsmTokenizer.from_pretrained(args_dict['arch'])   
    train_loader, val_loader, train_len, val_len = get_wds_loaders(rla_args, coordinator_params, gpu=None, shuffle_train=False, val_only=False, return_count=False)
    trained_model = trained_model.eval()

    esm_model = EsmForMaskedLM.from_pretrained('/data1/groups/keating_madry/huggingface/esm2_t30_150M_UR50D')
    base_lm_head = esm_model.lm_head
    lm_up_proj = True
    if lm_up_proj:
        lm_head = ReshapeBaseHead(base_lm_head)
    else:
        lm_head = EsmLMHead()
    lm_head = lm_head.cuda()

    if model_type == 'MLP':
        converter = ConverterModel(in_features=[320, 512, 512], out_features=[512, 512, 320], num_layers=3, dropout=0).to(device = dev)
    elif model_type == 'transformer':
        print(':)')
        converter = ConverterTransformerModel(d_model=320, nhead=4, num_encoder_layers=6, num_decoder_layers=1, dim_feedforward=512, dropout=0.0, batch_first=True, text_feat_type=text_feat_type, lm_head=lm_head).to(device=dev)
    elif model_type == 'encoder':
        print('X)')
        converter = ConverterTransformerEncoderModel(d_model=320, nhead=4, num_encoder_layers=6, dim_feedforward=512, dropout=0.0, text_feat_type=text_feat_type, lm_head=lm_head).to(device=dev)
    loss_fn = nn.CosineSimilarity(dim=1)
    loss_fn = torch.nn.MSELoss(reduction='none')

    converter_state = os.path.join(run_dir, 'net_best_checkpoint.pt')
    state_dict = torch.load(converter_state, map_location=dev)
    converter.load_state_dict(state_dict['state_dict'])

    torch.set_grad_enabled(False)
    converter = converter.eval()

    esm_model = EsmForMaskedLM.from_pretrained(esm_path)
    lm_head = esm_model.lm_head
    lm_head = lm_head.to('cuda:0')

    lm_head_base = copy.deepcopy(lm_head)
    lm_head_converter = copy.deepcopy(lm_head)

    base_optimizer = torch.optim.Adam(lm_head_base.parameters(), lr=1e-3, weight_decay=1e-4)
    converter_optimizer = torch.optim.Adam(lm_head_converter.parameters(), lr=1e-3, weight_decay=1e-4)
    new_optimizer = torch.optim.Adam(lm_head_converter.parameters(), lr=1e-3, weight_decay=1e-4)

    rla_args.batch_size=1
    train_loader, val_loader, train_len, val_len = get_wds_loaders(rla_args, coordinator_params, gpu=None, shuffle_train=False, val_only=False, return_count=False)

