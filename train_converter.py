import setuptools ## Need this for distutils bug
import src.models_and_optimizers as model_utils
import src.data_utils as data_utils
from types import SimpleNamespace
from clip_main import get_wds_loaders
from transformers import EsmTokenizer, EsmForMaskedLM
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
import esm as esmlib

def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class EsmLMHead(nn.Module):
    def __init__(self):
        super(EsmLMHead, self).__init__()
        self.dense = nn.Linear(in_features=640, out_features=640, bias=True)
        self.layer_norm = nn.LayerNorm(normalized_shape=(640,), eps=1e-5, elementwise_affine=True)
        self.decoder = nn.Linear(in_features=640, out_features=33, bias=False)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, x):
        return self.decoder(self.layer_norm(self.dense(x)))
    
class ReshapeBaseHead(nn.Module):
    def __init__(self, lm_head):
        super(ReshapeBaseHead, self).__init__()
        self.reshape_layer = nn.Linear(320, 640, bias=False)
        self.lm_head = lm_head
        
    def forward(self, x):
        x = self.reshape_layer(x)
        x = self.lm_head(x)
        return x

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
    
class ReshapeHead(nn.Module):
    def __init__(self, dim1, lm_head, dim2=640):
        super(ReshapeHead, self).__init__()
        self.reshape_layer = nn.Linear(dim1, dim2, bias=False)
        self.lm_head = lm_head
        
    def forward(self, x):
        x = self.reshape_layer(x)
        x = self.lm_head(x)
        return x
    
class ConverterTransformerEncoderModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, d_model=640, nhead=4, num_encoder_layers=2, dim_feedforward=512, dropout=0.0, text_feat_type='no_proj', batch_first=True, use_encoder=False, lm_head=None):
        super(ConverterTransformerEncoderModel, self).__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_encoder_layers=num_encoder_layers, dropout=dropout, batch_first=batch_first)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.shape_encoder = nn.Linear(320, d_model, bias=False)
        self.use_encoder = use_encoder
        self.decoder = nn.Linear(d_model, d_model)

        if text_feat_type in ['up_proj', 'esm']:
            self.reshaper = nn.Linear(320, 640, bias=True)
        self.text_feat_type = text_feat_type

        self.lm_head = lm_head

        for name, p in self.named_parameters():
            if p.dim() > 1 and 'lm_head' not in name:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def forward(self, src, tgt, use_mask=True):
        if self.use_encoder:
            src = self.shape_encoder(src)

        output = self.encoder(src) #mask=self.src_mask
        output = self.decoder(output)
        if self.text_feat_type in ['up_proj']:
            output = self.reshaper(output)

        if self.lm_head is not None:
            lm_out = self.lm_head(output.squeeze(0))
        else:
            lm_out = None
        return output.squeeze(0), lm_out
    
    def sample(self, src):
        if self.use_encoder:
            src = self.shape_encoder(src)

        output = self.encoder(src) #mask=self.src_mask
        output = self.decoder(output)
        if self.text_feat_type in ['up_proj']:
            output = self.reshaper(output)
        if self.lm_head is not None:
            lm_out = self.lm_head(output.squeeze(0))
        else:
            lm_out = None
        return output.squeeze(0), lm_out


class ConverterTransformerModel(nn.Module):
    def __init__(self, d_model=320, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0, batch_first=True, text_feat_type='no_proj', decoder=None, use_encoder=False, lm_head=None):
        super(ConverterTransformerModel, self).__init__()
        if decoder == 'linear':
            custom_decoder = torch.nn.Linear(320, 320)
            num_decoder_layers = 1
        else:
            custom_decoder = None
        self.transformer = torch.nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first, custom_decoder=custom_decoder)
        if text_feat_type in ['esm', 'up_proj']:
            self.reshaper = nn.Linear(d_model, 640, bias=False)
        self.text_feat_type = text_feat_type
        self.encoder = nn.Linear(320, d_model, bias=False)
        self.use_encoder = use_encoder 
        self.nhead = nhead
        self.lm_head = lm_head

        for name, p in self.named_parameters():
            if p.dim() > 1 and 'lm_head' not in name:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, use_mask=True):
        if self.use_encoder:
            src = self.encoder(src)
        if use_mask:
            # mask_attend = torch.ones(src.shape[1], src.shape[1], dtype=torch.bool, device='cuda:0')
            # for i in range(src.shape[1]):
            #     mask_attend[i, :i+1] = False
            # mask_attend = torch.zeros(src.shape[1], src.shape[1], dtype=torch.bool).to(device=src.device)
            # for i in range(src.shape[1]):
            #     mask_attend[i, i:] = True
                # if i == 0:
                #     mask_attend[i, i+1:] = True
                # else:
                #     mask_attend[i, 1:] = True #i:
            # mask_attend = mask_attend.unsqueeze(0).expand(self.nhead, src.shape[1], src.shape[1])
            size = src.shape[1]
            mask_attend = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
            mask_attend = mask_attend.float().masked_fill(mask_attend == 0, float('-inf')).masked_fill(mask_attend == 1, float(0.0)).to(device=src.device)
        else:
            mask_attend = None
        # mask_attend = ~mask_attend
        pred = self.transformer(src, tgt, tgt_mask=mask_attend).squeeze(0)
        if self.text_feat_type in ['up_proj', 'esm']:
            pred = self.reshaper(pred)

        if self.lm_head is not None:
            lm_out = self.lm_head(pred.squeeze(0))
        else:
            lm_out = None
        
        return pred.squeeze(0), lm_out

    def train_sample(self, src, real_tgt=None):
        if self.use_encoder:
            src = self.encoder(src)
        tgt = torch.zeros_like(src[:,1:]).to(device=src.device)
        tgt = F.pad(tgt, (0, 0, 0, 1))
        if real_tgt is not None:
            tgt[0, 1] = real_tgt[0]
        # mask_attend = torch.zeros(src.shape[1], src.shape[1], dtype=torch.bool).to(device=src.device)
        # for i in range(src.shape[1]):
        #     if i == 0:
        #         mask_attend[i, i+1:] = True
        #     else:
        #         mask_attend[i, i:] = True
        size = src.shape[1]
        mask_attend = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask_attend = mask_attend.float().masked_fill(mask_attend == 0, float('-inf')).masked_fill(mask_attend == 1, float(0.0)).to(device=src.device)
        prediction = []
        for i in range(1, src.shape[1]-1):
            tgt[0, i+1] = self.transformer(src, tgt, tgt_mask=mask_attend)[0, i]
            prediction.append(tgt[0, i])
        prediction = torch.stack(prediction, dim=0)
        return tgt, None
    
    def sample(self, src, real_tgt=None):
        if self.use_encoder:
            src = self.encoder(src)
        memory = self.transformer.encoder(src)
        tgt = torch.zeros(1, 1, src.shape[-1]).to(device=src.device)
        for i in range(src.shape[1]):
            size = tgt.shape[1]
            mask_attend = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
            mask_attend = mask_attend.float().masked_fill(mask_attend == 0, float('-inf')).masked_fill(mask_attend == 1, float(0.0)).to(device=src.device)
            if real_tgt is None:
                out = self.transformer.decoder(tgt, memory, tgt_mask=mask_attend)[:,-1].unsqueeze(1)
            else:
                out = self.transformer.decoder(real_tgt[:,:i+1], memory, tgt_mask=mask_attend)[:,-1].unsqueeze(1)
            tgt = torch.cat([tgt, out], dim=1)
        tgt = tgt[0,1:]
        if self.lm_head is not None:
            lm_out = self.lm_head(tgt)
        else:
            lm_out = None
        return tgt, lm_out

def mse_loss(pred, tgt):
    loss_fn = torch.nn.MSELoss(reduction='none')
    loss = loss_fn(pred, tgt)
    loss = torch.mean(loss, dim=-1)
    return loss

def run_iter(batch_iter, len_iter, converter, optimizer, loss_fn, dir_loss_fn, trained_model, tokenizer, model_type, text_feat_type, dev, esm, batch_converter, grad, teacher_forcing=True, seq_layer=None, dir_loss_weight=0.0, percent_error_weight=0.0, nsr_loss_weight=0.0, lm_head_loss=None):
    losses = []
    dir_losses = []
    percent_errors = []
    nsrs = []
    with torch.autograd.set_detect_anomaly(True):
        for i, batch in enumerate(batch_iter):
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

                    seqs = batch[0]['string_sequence']
                    text_inp = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024+2)
                    text_inp['position_ids'] = batch[0]['pos_embs'][0]
                    text_inp = {k: v.to(dev) for k, v in text_inp.items()}
                    seq_640 = trained_model.get_text_features_no_proj(text_inp)
                    seq_640, _, _ = data_utils.postprocess_text_features(text_features=seq_640, inp_dict=text_inp, tokenizer=tokenizer, placeholder_mask=batch[0]['placeholder_mask'][0])
                    seq_640 = seq_640.squeeze(0)

            esm_data = [('prot', batch[0]['string_sequence'][0])]
            _, _, batch_tokens = batch_converter(esm_data)
            batch_tokens = batch_tokens.to(args.dev)
            target = batch_tokens[0,1:-1]
            if text_feat_type == 'esm':
                results = esm(batch_tokens, repr_layers=[30], return_contacts=False)
                seq_for_loss = results['representations'][30].squeeze(0)[1:-1]
                seq_for_loss = seq_for_loss[batch[0]['placeholder_mask'][0][0]]

            struct_embeddings = output_dict['gnn'][output_dict['coord_mask_with_burn_in']]
            # if batch[0]['pdb_id'][0] == '3u6g':
            #     print(seq_embeddings)
            #     print('------')
            #     print(struct_embeddings)
            #     raise ValueError
            if seq_embeddings.shape[0] != struct_embeddings.shape[0]:
                print(i, batch[0]['pdb_id'], seq_embeddings.shape, struct_embeddings.shape)
                raise ValueError
            mask = output_dict['seq_mask_no_burn_in'].flatten().to(device=dev)

            if seq_layer is None:
                seq_for_model = seq_embeddings
                seq_for_loss = seq_embeddings
            else:
                seq_for_model = seq_layer(seq_640)
                seq_for_loss = seq_layer(seq_640)
            if text_feat_type in ['up_proj', '640']:
                seq_for_loss = seq_640
                seq_for_model = seq_640
            if grad:
                torch.set_grad_enabled(True)
                # seq_for_model = seq_for_loss
                if text_feat_type == 'esm':
                    seq_for_model = seq_for_loss
                converter.train()
            else:
                torch.set_grad_enabled(False)
                # seq_for_model = torch.zeros_like(seq_embeddings)
                # seq_for_model = seq_for_loss
                if text_feat_type == 'esm':
                    seq_for_model = seq_for_loss
                converter.eval()

            if model_type == 'MLP':
                struct2seq_embeddings = converter(struct_embeddings)
            elif model_type == 'transformer':
                # Pad with 0s for start/end token
                # struct_embeddings = F.pad(struct_embeddings, (0, 0, 1, 0))
                seq_for_model = F.pad(seq_for_model, (0, 0, 1, 0))
                seq_for_model = seq_for_model[:-1]
                if teacher_forcing:
                    struct2seq_embeddings, lm_out = converter(struct_embeddings.unsqueeze(0), seq_for_model.unsqueeze(0), use_mask=True)
                else:
                    struct2seq_embeddings, lm_out = converter.sample(struct_embeddings.unsqueeze(0))
                # if not use_sample:
                #     struct2seq_embeddings = converter(struct_embeddings.unsqueeze(0), seq_for_model.unsqueeze(0)).squeeze(0)
                # else:
                #     struct2seq_embeddings = converter.train_sample(struct_embeddings.unsqueeze(0)).squeeze(0)
                # struct2seq_embeddings = struct2seq_embeddings
            loss = loss_fn(seq_for_loss, struct2seq_embeddings)
            loss = torch.sum(loss * mask) / torch.sum(mask)
            jloss = copy.deepcopy(loss.item())

            

            dir_loss = dir_loss_fn(seq_for_loss, struct2seq_embeddings)
            dir_loss = -1 * torch.sum(dir_loss * mask) / torch.sum(mask)

            percent_error = torch.mean((torch.abs((seq_for_loss - struct2seq_embeddings)) / struct2seq_embeddings), dim=-1)
            percent_error = torch.sum(percent_error * mask) / torch.sum(mask) / 100

            lm_pred = torch.softmax(lm_out, dim=-1)
            lm_loss = lm_head_loss(lm_pred, target)
            lm_loss = torch.sum(lm_loss * mask) / torch.sum(mask)
            pred_tokens = torch.argmax(lm_pred, dim=-1)
            nsr = (pred_tokens == target).sum() / target.numel()

            sum_loss = loss
            if dir_loss_weight > 0:
                sum_loss += dir_loss
            if percent_error_weight > 0:
                sum_loss += percent_error
            if nsr_loss_weight < 0:
                sum_loss = lm_loss
            elif nsr_loss_weight > 0:
                sum_loss += lm_loss



            if grad:
                optimizer.zero_grad()
                # loss.backward()
                # dir_loss.backward()
                sum_loss.backward()
                optimizer.step()
            losses.append(jloss)
            dir_losses.append(dir_loss.item())
            percent_errors.append(percent_error.item())
            nsrs.append(nsr.item())
    return np.mean(losses), np.mean(dir_losses), np.mean(percent_errors), np.mean(nsrs)

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
    parser.add_argument('--d_model', help='Model dimension', default=320, type=int)
    parser.add_argument('--n_head', help='Number of attention heads', default=4, type=int)
    parser.add_argument('--dim_feed', help='Feedforward dimension', default=512, type=int)
    parser.add_argument('--teacher_forcing', help='Whether to train with teacher forcing', default=True, type=bool)
    parser.add_argument('--use_seq_layer', help='Whether to predict seq layer', default=False)
    parser.add_argument('--dir_loss_weight', help='Weight given to cosine similarity loss', default=0.0, type=float)
    parser.add_argument('--percent_error_weight', help='Weight given to percent error loss', default=0.0, type=float)
    parser.add_argument('--nsr_loss_weight', help='Weight given to NSR loss', default=0.0, type=float)
    parser.add_argument('--lm_head_cpt', help='Language model head checkpoint', default='/home/gridsan/fbirnbaum/joint-protein-embs/lm_head_runs/multichain_base_new_converter_reshape/net_best_checkpoint_converter.pt', type=str)
    parser.add_argument('--lm_up_proj', help='Whether to reshape lm dim', default=False, type=bool)
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
    print("Teacher forcing: ", args.teacher_forcing)
    print("Epochs: ", args.epochs)
    print("Use seq layer: ", args.use_seq_layer)
    print("Dir loss weight: ", args.dir_loss_weight)
    print("Percent error weight: ", args.percent_error_weight)
    print("NSR loss weight: ", args.nsr_loss_weight)
    print("LM head cpt: ", args.lm_head_cpt)
    print("LM up proj: ", args.lm_up_proj)
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
    
    if args.loss_fn == 'MSE':
        loss_fn = mse_loss
    elif args.loss_fn == 'cosine':
        loss_fn = nn.CosineSimilarity(dim=1)
    dir_loss_fn = nn.CosineSimilarity(dim=1)

    esm_model = EsmForMaskedLM.from_pretrained(args.esm_path)
    base_lm_head = esm_model.lm_head
    if args.lm_up_proj:
        lm_head = ReshapeBaseHead(base_lm_head)
    else:
        lm_head = EsmLMHead()
    if args.dev == 'cuda:0':
        lm_head = lm_head.cuda()
    
    if args.lm_up_proj:
        state_dict = torch.load(args.lm_head_cpt, map_location=args.dev)
        lm_head.load_state_dict(state_dict['state_dict'])
    lm_head_loss = torch.nn.NLLLoss(reduction='none')

    if args.model_type == 'MLP':
        converter = ConverterModel(in_features=[320, 512, 512], out_features=[512, 512, 320], num_layers=3, dropout=0).to(device=args.dev)
    elif args.model_type == 'transformer':
        if args.decoder == 'autoregressive':
            converter = ConverterTransformerModel(d_model=args.d_model, nhead=args.n_head, num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers, dim_feedforward=args.dim_feed, dropout=args.dropout, batch_first=True, text_feat_type=args.text_feat_type, decoder=args.decoder, lm_head=lm_head).to(device=args.dev)
        elif args.decoder == 'linear':
            converter = ConverterTransformerEncoderModel(d_model=args.d_model, nhead=args.n_head, num_encoder_layers=args.num_encoder_layers, dim_feedforward=args.dim_feed, dropout=args.dropout, text_feat_type=args.text_feat_type, batch_first=True, lm_head=lm_head).to(device=args.dev)

    optimizer = torch.optim.Adam(converter.parameters(), lr=args.lr, weight_decay=args.regularization)

    writer = SummaryWriter(log_dir=os.path.join(args.run_dir, 'tensorboard'))
    training_curves = {"train_loss": [], "val_loss": [], "val_no_tf_loss": [], "train_loss_dir": [], "val_loss_dir": [], "val_no_tf_loss_dir": [], "train_pe": [], "val_pe": [], "val_no_tf_pe": [], "train_nsr": [], "val_nsr": [], "val_no_tf_nsr": []}
    best_val_loss = None
    best_no_tf_loss = None
    epochs_since_improvement = 0

    esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm = esm.to(args.dev)

    if args.use_seq_layer:
        esm_small, _ = esmlib.pretrained.esm2_t6_8M_UR50D()
        lm_head_small = esm_small.lm_head
        seq_layer = ReshapeHead(640, lm_head_small, 320)
        checkpoint_state = torch.load('/home/gridsan/fbirnbaum/joint-protein-embs/lm_head_runs/test_reshape_checkpoint_base.pt')
        seq_layer.load_state_dict(checkpoint_state['state_dict'])
        seq_layer = seq_layer.reshape_layer.cuda()
    else:
        seq_layer = None

    for epoch in range(args.epochs):
        
        # Train iter
        train_loss, train_loss_dir, train_pe, train_nsr = run_iter(train_loader, train_len, converter, optimizer, loss_fn, dir_loss_fn, trained_model, tokenizer, args.model_type, args.text_feat_type, args.dev, esm, batch_converter, grad=True, seq_layer=seq_layer, dir_loss_weight=args.dir_loss_weight, percent_error_weight=args.percent_error_weight, nsr_loss_weight=args.nsr_loss_weight, lm_head_loss=lm_head_loss) #use_sample=(not args.teacher_forcing)

        # Val iter
        val_loss, val_loss_dir, val_pe, val_nsr = run_iter(val_loader, val_len, converter, optimizer, loss_fn, dir_loss_fn, trained_model, tokenizer, args.model_type, args.text_feat_type, args.dev, esm, batch_converter, grad=False, seq_layer=seq_layer, dir_loss_weight=args.dir_loss_weight, percent_error_weight=args.percent_error_weight, nsr_loss_weight=args.nsr_loss_weight,lm_head_loss=lm_head_loss)

        # No TF Val iter
        if epoch % 5 == 0:
            no_tf_loss, no_tf_loss_dir, no_tf_pe, no_tf_nsr = run_iter(val_loader, val_len, converter, optimizer, loss_fn, dir_loss_fn, trained_model, tokenizer, args.model_type, args.text_feat_type, args.dev, esm, batch_converter, grad=False, teacher_forcing=False, seq_layer=seq_layer, dir_loss_weight=args.dir_loss_weight, percent_error_weight=args.percent_error_weight, nsr_loss_weight=args.nsr_loss_weight, lm_head_loss=lm_head_loss)

        # Upkeep        
        progress.update(1)
        progress.refresh()
        progress.set_description_str(f'train loss {train_loss} | val loss {val_loss} | no tf loss {no_tf_loss} | train loss dir {train_loss_dir} | val loss dir {val_loss_dir} | no tf loss dir {no_tf_loss_dir} | train percent error {train_pe} | val percent error {val_pe} | no tf percent error {no_tf_pe} | train nsr {train_nsr} | val nsr {val_nsr} | no tf nsr {no_tf_nsr}')
        writer.add_scalar('train loss', train_loss, epoch)
        writer.add_scalar('val loss', val_loss, epoch)
        writer.add_scalar('val no tf loss', no_tf_loss, epoch)
        writer.add_scalar('train loss dir', train_loss_dir, epoch)
        writer.add_scalar('val loss dir', val_loss_dir, epoch)
        writer.add_scalar('val no tf loss dir', no_tf_loss_dir, epoch)
        writer.add_scalar('train percent error', train_pe, epoch)
        writer.add_scalar('val percent error', val_pe, epoch)
        writer.add_scalar('val no tf percent error', no_tf_pe, epoch)
        writer.add_scalar('train nsr', train_nsr, epoch)
        writer.add_scalar('val nsr', val_nsr, epoch)
        writer.add_scalar('val no tf nsr', no_tf_nsr, epoch)
        training_curves["train_loss"].append(train_loss)
        training_curves["val_loss"].append(val_loss)
        training_curves["val_no_tf_loss"].append(no_tf_loss)
        training_curves["train_loss_dir"].append(train_loss_dir)
        training_curves["val_loss_dir"].append(val_loss_dir)
        training_curves["val_no_tf_loss_dir"].append(no_tf_loss_dir)
        training_curves["train_pe"].append(train_pe)
        training_curves["val_pe"].append(val_pe)
        training_curves["val_no_tf_pe"].append(no_tf_pe)
        training_curves["train_nsr"].append(train_nsr)
        training_curves["val_nsr"].append(val_nsr)
        training_curves["val_no_tf_nsr"].append(no_tf_nsr)
        # Save a state checkpoint
        checkpoint_state = {
            'epoch': epoch,
            'state_dict': converter.state_dict(),
            'val_loss': val_loss,
            'optimizer_state': optimizer.state_dict(),
            'training_curves': training_curves
        }
        torch.save(checkpoint_state, os.path.join(args.run_dir, 'net_last_checkpoint.pt'))

        # Early stopping
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            torch.save(checkpoint_state, os.path.join(args.run_dir, 'net_best_checkpoint.pt'))
        else:
            epochs_since_improvement += 1

        if best_no_tf_loss is None or no_tf_loss < best_no_tf_loss:
            best_no_tf_loss = no_tf_loss
            torch.save(checkpoint_state, os.path.join(args.run_dir, 'net_best_notf_checkpoint'))

        if args.early_stopping and epochs_since_improvement >= 15:
            print("Early stopping")
            break