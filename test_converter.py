import setuptools ## Need this for distutils bug
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
import esm as esmlib

def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class ConverterModel(nn.Module):
    def __init__(self, in_features, out_features, num_layers, activation_layers='relu', dropout=0):
        super(ConverterModel, self).__init__()
        self.activation_layers = activation_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_features[i], out_features[i]))
        self.dropout_prob = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout, inplace=False)
        self.prob_activation = torch.nn.Sigmoid()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            
    def forward(self, x):
        for i, layer in enumerate(self.layers):  
            x = layer(x)
            if self.activation_layers == 'relu':
                x = F.relu(x)
            else:
                x = gelu(x)
        if self.dropout_prob > 0:
            x = self.dropout(x)
        x = self.prob_activation(x).squeeze(-1)
        return x
    
class ConverterTransformerEncoderModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, d_model=640, nhead=4, num_encoder_layers=2, dim_feedforward=512, dropout=0.0, text_feat_type='no_proj', batch_first=True):
        super(ConverterTransformerEncoderModel, self).__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_encoder_layers=num_encoder_layers, dropout=dropout, batch_first=batch_first)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.shape_encoder = nn.Linear(320, d_model, bias=False)
        self.decoder = nn.Linear(d_model, d_model)

        # if text_feat_type in ['no_proj', 'esm']:
        #     self.reshaper = nn.Linear(320, 640, bias=False)
        # self.text_feat_type = text_feat_type

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def forward(self, src, tgt, has_mask=True):
        src = self.shape_encoder(src)
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        output = self.encoder(src) #mask=self.src_mask
        output = self.decoder(output)
        return output


class ConverterTransformerModel(nn.Module):
    def __init__(self, d_model=320, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0, batch_first=True, text_feat_type='proj', decoder=None):
        super(ConverterTransformerModel, self).__init__()
        if decoder == 'linear':
            custom_decoder = torch.nn.Linear(320, 320)
            num_decoder_layers = 1
        else:
            custom_decoder = None
        self.transformer = torch.nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first, custom_decoder=custom_decoder)
        if text_feat_type in ['no_proj', 'esm']:
            self.reshaper = nn.Linear(d_model, 640, bias=False)
        self.text_feat_type = text_feat_type
        self.encoder = nn.Linear(320, d_model, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, use_mask=True):
        src = self.encoder(src)
        if use_mask:
            # mask_attend = torch.ones(src.shape[1], src.shape[1], dtype=torch.bool, device='cuda:0')
            # for i in range(src.shape[1]):
            #     mask_attend[i, :i+1] = False
            mask_attend = torch.zeros(src.shape[1], src.shape[1], dtype=torch.bool).to(device=src.device)
            for i in range(src.shape[1]):
                if i == 0:
                    mask_attend[i, i+1:] = True
                else:
                    mask_attend[i, i:] = True
        else:
            mask_attend = None
            
        pred = self.transformer(src, tgt, tgt_mask=mask_attend).squeeze(0)
        # if self.text_feat_type in ['no_proj', 'esm']:
        #     pred = self.reshaper(pred)
        return pred

    def sample(self, src, real_tgt=None):
        src = self.encoder(src)
        tgt = torch.rand_like(src[:,1:-1]).to(device=src.device)
        tgt = F.pad(tgt, (0, 0, 1, 1))
        mask_attend = torch.zeros(src.shape[1], src.shape[1], dtype=torch.bool).to(device=src.device)
        for i in range(src.shape[1]):
            if i == 0:
                mask_attend[i, i+1:] = True
            else:
                mask_attend[i, i:] = True
        for i in range(1, src.shape[1]-1):
            if real_tgt is not None and i > 1:
                pred_tgt = copy.deepcopy(tgt[0,i-1])
                tgt[0, i-1] = real_tgt[i-1]
            tgt[0, i] = self.transformer(src, tgt, tgt_mask=mask_attend)[0, i]
            if real_tgt is not None and i > 1:
                tgt[0, i-1] = pred_tgt
        return tgt

def mse_loss(pred, tgt):
    loss_fn = torch.nn.MSELoss(reduction='none')
    loss = loss_fn(pred, tgt)
    loss = torch.mean(loss, dim=-1)
    return loss

def run_iter(batch_iter, len_iter, converter, optimizer, loss_fn, trained_model, tokenizer, model_type, text_feat_type, dev, esm, batch_converter, grad):
    losses = []
    val_losses = []
    for i, batch in enumerate(batch_iter):
        if i > 10:
            break
        if len(batch[0]['string_sequence'][0]) < 30:
            continue
        # for b in range(batch[1]['coords'][0].shape[0]):
        #     batch[1]['coords'][0][b] -= torch.mean(batch[1]['coords'][0][b], dim=[0,1])
        # if len(batch[0]['string_sequence'][0]) != 228:
        #     continue
        chain_lens = torch.zeros(batch[1]['coords'][0].shape[1]).to(device = batch[1]['coords'][0].device)
        chain_lens[batch[1]['seq_lens'][0][0]:] = 1
        chain_lens_mask = torch.ones(batch[1]['coords'][0].shape[1]).unsqueeze(0).to(dtype=torch.bool, device = batch[1]['coords'][0].device)    
        batch[1]['chain_lens'] = [chain_lens.unsqueeze(0), chain_lens_mask]
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                output_dict = get_text_and_image_features(trained_model, tokenizer, batch, pdb=None, weight_dists=False, seq_mask=None, focus=False, top_k=30, struct_mask=None, 
                                                          remove_far=False, threshold=1, dev=dev)
                seq_embeddings = output_dict['text'].squeeze(0)

                if text_feat_type == 'no_proj':
                    seqs = batch[0]['string_sequence']
                    text_inp = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024+2)
                    text_inp['position_ids'] = batch[0]['pos_embs'][0]
                    text_inp = {k: v.to(dev) for k, v in text_inp.items()}
                    seq_for_loss = trained_model.get_text_features_no_proj(text_inp)
                    seq_for_loss, _, _ = data_utils.postprocess_text_features(text_features=seq_for_loss, inp_dict=text_inp, tokenizer=tokenizer, placeholder_mask=batch[0]['placeholder_mask'][0])
                    seq_for_loss = seq_for_loss.squeeze(0)
                else:
                    seq_for_loss = output_dict['text'].squeeze(0)
        if text_feat_type == 'esm':
            esm_data = [('prot', batch[0]['string_sequence'][0])]
            _, _, batch_tokens = batch_converter(esm_data)
            batch_tokens = batch_tokens.to(args.dev)
            results = esm(batch_tokens, repr_layers=[30], return_contacts=False)
            seq_for_loss = results['representations'][30].squeeze(0)[1:-1]
            seq_for_loss = seq_for_loss[batch[0]['placeholder_mask'][0][0]]

        struct_embeddings = output_dict['gnn'][output_dict['coord_mask_with_burn_in']]

        if seq_embeddings.shape[0] != struct_embeddings.shape[0]:
            print(i, batch[0]['pdb_id'], seq_embeddings.shape, struct_embeddings.shape)
            raise ValueError
        mask = output_dict['seq_mask_no_burn_in'].flatten().to(device=dev)

        seq_for_model = seq_for_loss

        if model_type == 'MLP':
            struct2seq_embeddings = converter(struct_embeddings)
        elif model_type == 'transformer':
            # Pad with 0s for start/end token
            struct_embeddings = F.pad(struct_embeddings, (0, 0, 1, 1))
            seq_for_model = F.pad(seq_for_model, (0, 0, 1, 1))
            struct2seq_embeddings_sample = converter.sample(struct_embeddings.unsqueeze(0), real_tgt=seq_for_model).squeeze(0)
            struct2seq_embeddings_sample = struct2seq_embeddings_sample[1:-1]
            struct2seq_embeddings = converter(struct_embeddings.unsqueeze(0), seq_for_model.unsqueeze(0)).squeeze(0)
            struct2seq_embeddings = struct2seq_embeddings[1:-1]
        print(struct2seq_embeddings[0][:20])
        print('-----')
        print(struct2seq_embeddings_sample[0][:20])
        loss = loss_fn(seq_for_loss, struct2seq_embeddings)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        losses.append(loss.item())

        loss = loss_fn(seq_for_loss, struct2seq_embeddings_sample)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        val_losses.append(loss.item())
        print('+++++')

    return np.mean(losses), np.mean(val_losses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train converter!')
    parser.add_argument('--run_dir', help='Directory to save training results', required=True)
    parser.add_argument('--train_wds', help='Train dataset.', required=True)
    parser.add_argument('--val_wds', help='Val dataset.', required=True)
    parser.add_argument('--epochs', help='Number of train epochs', default=100, type=int)
    parser.add_argument('--lr', help='Learning rate', default=1e-5, type=float)
    parser.add_argument('--regularization', help='Regularization', default=1e-3, type=float)
    parser.add_argument('--dropout', help='Dropout', default=0.0, type=float)
    parser.add_argument('--batch_size', help='Batch size', default=1, type=int)
    parser.add_argument('--data_root', help='Path to data dir', default='/data1/groups/keating_madry/wds/')
    parser.add_argument('--model_root', help='Path to model dir', default='/data1/groups/keating_madry/runs/new_blacklist', type=str)
    parser.add_argument('--esm_path', help='Path to model dir', default='/data1/groups/keating_madry/huggingface/esm2_t30_150M_UR50D', type=str)
    parser.add_argument('--config', help='COORDinator config', default='/home/gridsan/fbirnbaum/joint-protein-embs/terminator_configs/coordinator_broken_merge.json', type=str)
    parser.add_argument('--model_type', help='Type of converter model', default='MLP', type=str)
    parser.add_argument('--text_feat_type', help='Type of seq features to use', default='proj', type=str)
    parser.add_argument('--loss_fn', help='Loss function to use', default='MSE', type=str)
    parser.add_argument('--dev', help='Device to train on', default='cuda:0', type=str)
    parser.add_argument('--early_stopping', help='Whether to use early stopping', default=False, type=bool)
    parser.add_argument('--decoder', help='Decoder type', default='autoregressive', type=str)
    parser.add_argument('--num_encoder_layers', help='Number of encoder layers', default=2, type=int)
    parser.add_argument('--num_decoder_layers', help='Number of decoder layers', default=2, type=int)
    parser.add_argument('--d_model', help='Model dimension', default=640, type=int)
    parser.add_argument('--n_head', help='Number of attention heads', default=4, type=int)
    parser.add_argument('--dim_feed', help='Feedforward dimension', default=512, type=int)
    args = parser.parse_args()

    print("Run dir: ", args.run_dir)
    print("Train wds: ", args.train_wds)
    print("Val wds: ", args.val_wds)
    print("Learning rate: ", args.lr)
    print("Regularization: ", args.regularization)
    print("Dropout: ", args.dropout)
    print("Batch size: ", args.batch_size)
    print("Data root: ", args.data_root)
    print("Model root: ", args.model_root)
    print("Esm path: ", args.esm_path)
    print("Config: ", args.config)
    print("Model type: ", args.model_type)
    print("Text feat type: ", args.text_feat_type)
    print("Loss function: ", args.loss_fn)
    print("Early stopping: ", args.early_stopping)
    print("Decoder: ", args.decoder)
    print("Num encoder layers: ", args.num_encoder_layers)
    print("Num decoder layers: ", args.num_decoder_layers)
    print("Model dimension: ", args.d_model)
    print("Num heads: ", args.n_head)
    print("Feedfoward dim: ", args.dim_feed)
    print("Dev: ", args.dev)

    model_dir = "version_0/" 

    CLIP_MODE = False

    ROOT = args.model_root

    root_path = os.path.join(ROOT, model_dir)
    path = os.path.join(root_path, "checkpoints/checkpoint_best.pt")
    data_root = args.data_root
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
    args_dict['train_wds_path'] = args.val_wds ## args.train_wds
    args_dict['val_wds_path'] = args.val_wds
    args_dict['batch_size'] = args.batch_size
    args_dict['blacklist_file'] = ''
    for k in backwards_compat.keys():
        if k not in args_dict:
            args_dict[k] = backwards_compat[k]
    rla_args = SimpleNamespace(**args_dict)
    rla_args.coordinator_hparams = args.config

    coordinator_params = data_utils.get_coordinator_params(rla_args.coordinator_hparams)
    coordinator_params['num_positional_embeddings'] = rla_args.gnn_num_pos_embs
    coordinator_params['zero_out_pos_embs']= rla_args.gnn_zero_out_pos_embs
    coordinator_params['clip_mode'] = True
    

    args_dict['arch']= args.esm_path
    trained_model = model_utils.load_model(path, args_dict['arch'], args.dev)
    tokenizer = EsmTokenizer.from_pretrained(args_dict['arch'])   
    train_loader, val_loader, train_len, val_len = get_wds_loaders(rla_args, coordinator_params, gpu=None, shuffle_train=False, val_only=False, return_count=False)
    trained_model = trained_model.eval()
    # train_len = 1
    # val_len = 1

    progress = tqdm(total=args.epochs)
    if args.model_type == 'MLP':
        converter = ConverterModel(in_features=[320, 512, 512], out_features=[512, 512, 320], num_layers=3, dropout=0).to(device=args.dev)
    elif args.model_type == 'transformer':
        if args.decoder == 'autoregressive':
            converter = ConverterTransformerModel(d_model=args.d_model, nhead=args.n_head, num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers, dim_feedforward=args.dim_feed, dropout=args.dropout, batch_first=True, text_feat_type=args.text_feat_type, decoder=args.decoder).to(device=args.dev)
        elif args.decoder == 'linear':
            converter = ConverterTransformerEncoderModel(d_model=args.d_model, nhead=args.n_head, num_encoder_layers=args.num_encoder_layers, dim_feedforward=args.dim_feed, dropout=args.dropout, text_feat_type=args.text_feat_type, batch_first=True).to(device=args.dev)

    checkpoint_state = torch.load(os.path.join(args.run_dir, 'net_best_checkpoint.pt'))
    converter.load_state_dict(checkpoint_state['state_dict'])

    if args.loss_fn == 'MSE':
        loss_fn = mse_loss
    elif args.loss_fn == 'cosine':
        loss_fn = nn.CosineSimilarity(dim=1)
    optimizer = torch.optim.Adam(converter.parameters(), lr=args.lr, weight_decay=args.regularization)

    if args.text_feat_type == 'esm':
        esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        esm = esm.to(args.dev)
    else:
        esm = None
        batch_converter = None

    torch.set_grad_enabled(False)
    converter.train()
    converter.eval()

    for epoch in range(args.epochs):
        
        # Val iter
        val_loss, val_loss_val = run_iter(val_loader, val_len, converter, optimizer, loss_fn, trained_model, tokenizer, args.model_type, args.text_feat_type, args.dev, esm, batch_converter, grad=False)

        # Upkeep        
        progress.update(1)
        progress.refresh()
        progress.set_description_str(f'val loss tf {val_loss} | val loss autoregressive {val_loss_val}')