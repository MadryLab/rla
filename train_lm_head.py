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

class EsmLMHead(nn.Module):
    def __init__(self, dmodel):
        super(EsmLMHead, self).__init__()
        self.dense = nn.Linear(in_features=dmodel, out_features=dmodel, bias=True)
        self.layer_norm = nn.LayerNorm(normalized_shape=(dmodel,), eps=1e-5, elementwise_affine=True)
        self.decoder = nn.Linear(in_features=dmodel, out_features=33, bias=False)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, x):
        return self.decoder(self.layer_norm(self.dense(x)))
    
    
class EsmLMHead_Large(nn.Module):
    def __init__(self, dmodel1, dmodel2):
        super(EsmLMHead_Large, self).__init__()
        self.dense = nn.Linear(in_features=dmodel1, out_features=dmodel1, bias=True)
        self.layer_norm = nn.LayerNorm(normalized_shape=(dmodel1,), eps=1e-5, elementwise_affine=True)
        self.decoder = nn.Linear(in_features=dmodel1, out_features=33, bias=False)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, x):
        return self.decoder(self.layer_norm(self.dense(x)))
    
class ReshapeHead(nn.Module):
    def __init__(self, dim1, lm_head, dim2=640):
        super(ReshapeHead, self).__init__()
        self.reshape_layer = nn.Linear(dim1, dim2, bias=False)
        self.lm_head = lm_head
        
    def forward(self, x):
        x = self.reshape_layer(x)
        x = self.lm_head(x)
        return x
    
def run_iter(batch, trained_model, tokenizer, converter, esm, lm_head_base, lm_head_converter, lm_head_new, base_optimizer, converter_optimizer, new_optimizer, desired_mse=0, grad=True):
    
    chain_lens = torch.zeros(batch[1]['coords'][0].shape[1]).to(device = batch[1]['coords'][0].device)
    chain_lens[batch[1]['seq_lens'][0][0]:] = 1
    chain_lens_mask = torch.ones(batch[1]['coords'][0].shape[1]).unsqueeze(0).to(dtype=torch.bool, device = batch[1]['coords'][0].device)    
    batch[1]['chain_lens'] = [chain_lens.unsqueeze(0), chain_lens_mask]
    criterion = torch.nn.NLLLoss(reduction='none')
    with torch.no_grad():
        with autocast(dtype=torch.float16):
            output_dict = get_text_and_image_features(trained_model, tokenizer, batch, pdb=None, weight_dists=False, seq_mask=None, focus=False, top_k=30, struct_mask=None, 
                                                          remove_far=False, threshold=1, dev='cuda:0')
        seq_embeddings = output_dict['text'].squeeze(0)
        seqs = batch[0]['string_sequence']
        text_inp = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024+2)
        text_inp['position_ids'] = batch[0]['pos_embs'][0]
        text_inp = {k: v.to('cuda:0') for k, v in text_inp.items()}
        base_seq_in = trained_model.get_text_features_no_proj(text_inp)
        base_seq_in, _, _ = data_utils.postprocess_text_features(text_features=base_seq_in, inp_dict=text_inp, tokenizer=tokenizer, placeholder_mask=batch[0]['placeholder_mask'][0])
        base_seq_in = base_seq_in.squeeze(0)

        struct_embeddings = output_dict['gnn'][output_dict['coord_mask_with_burn_in']]
        seq_embeddings = output_dict['text'].squeeze(0)
        # seq_embeddings = F.pad(base_seq_in, (0, 0, 1, 0))[:-1]
        base_seq_in_cheat = F.pad(base_seq_in, (0, 0, 1, 0))[:-1]
        seq_for_converter = F.pad(seq_embeddings, (0, 0, 1, 0))[:-1]

        ind = batch[0]['placeholder_mask'][0][0].cpu().numpy()
        esm_seq = "".join(np.array(list(batch[0]['string_sequence'][0]))[ind])
        esm_data = [('prot', esm_seq)]
        _, _, batch_tokens = batch_converter(esm_data)
        batch_tokens = batch_tokens.to('cuda:0')
        esm_results = esm(batch_tokens, repr_layers=[30], return_contacts=False)
        esm_seq_in = esm_results['representations'][30].squeeze(0)[1:-1]
        # esm_seq_in = esm_seq_in[batch[0]['placeholder_mask'][0][0]]
        target = batch_tokens[0,1:-1]
        converter_seq_in, _ = converter.sample(struct_embeddings.unsqueeze(0))
        # converter_seq_in = converter(struct_embeddings.unsqueeze(0), base_seq_in_cheat.unsqueeze(0)).squeeze(0)
        # converter_seq_in = converter(struct_embeddings.unsqueeze(0), seq_for_converter.unsqueeze(0), use_mask=True).squeeze(0)

        # new_seq_in = copy.deepcopy(converter_seq_in)

    noise_std = torch.sqrt(torch.tensor(desired_mse))
    noise = torch.normal(mean=0.0, std=noise_std, size=seq_embeddings.size()).to('cuda:0')
    seq_embeddings = seq_embeddings + noise
    # try:
    # base_pred = torch.softmax(lm_head_base(seq_embeddings), dim=-1)
    # converter_pred = torch.softmax(lm_head_converter(converter_seq_in), dim=-1)
    base_pred = torch.softmax(lm_head_base(base_seq_in), dim=-1)
    converter_pred = torch.softmax(lm_head_converter(converter_seq_in), dim=-1)
    # new_pred = torch.softmax(lm_head_new(esm_seq_in), dim=-1)
    base_loss = criterion(base_pred, target).mean()
    converter_loss = criterion(converter_pred, target).mean()


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
    # base_nsr = torch.Tensor([0])
    # base_loss = torch.Tensor([0])
    new_nsr = torch.Tensor([0])
    new_loss = torch.Tensor([0])
    return base_loss.item(), base_nsr.item(), converter_loss.item(), converter_nsr.item(), new_loss.item(), new_nsr.item()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train converter!')
    parser.add_argument('--out_dir', help='Dir to save output', required=True)
    parser.add_argument('--run_dir', help='Converter model', default='/home/gridsan/fbirnbaum/joint-protein-embs/converter_runs/ingraham_transformer_dec_1_lr_5_reg_3_larger_drop_0_mse_noproj_eval_tf_e1000_tracknotf_noenc_save')
    parser.add_argument('--val_wds', help='Val wds path', default='multichain_clip_val.wds')
    parser.add_argument('--train_wds', help='Train wds path', default='multichain_clip_train.wds')
    parser.add_argument('--data_root', help='wds dir', default='/data1/groups/keating_madry/wds/')
    parser.add_argument('--rla_root', help='model dir', default='/data1/groups/keating_madry/runs/new_blacklist')
    parser.add_argument('--esm_path', help='Path to ESM', default='/data1/groups/keating_madry/huggingface/esm2_t30_150M_UR50D')
    parser.add_argument('--config', help='Path to model config file', default='/home/gridsan/fbirnbaum/joint-protein-embs/terminator_configs/coordinator_broken_merge.json')
    parser.add_argument('--model_type', help='Type of converter', default='transformer')
    parser.add_argument('--dev', help='Device to train on', default='cuda:0')
    parser.add_argument('--text_feat_type', help='Text features type', default='no_proj', type=str)
    args = parser.parse_args()

    print(args)

    esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm = esm.cuda()

    model_dir = "version_0/" 
    CLIP_MODE = False
    ROOT = args.rla_root
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
    args_dict['data_root'] = args.data_root
    args_dict['train_wds_path'] = args.train_wds
    args_dict['val_wds_path'] = args.val_wds
    args_dict['batch_size'] = 1
    for k in backwards_compat.keys():
        if k not in args_dict:
            args_dict[k] = backwards_compat[k]
    rla_args = SimpleNamespace(**args_dict)
    rla_args.coordinator_hparams = args.config

    coordinator_params = data_utils.get_coordinator_params(rla_args.coordinator_hparams)
    coordinator_params['num_positional_embeddings'] = rla_args.gnn_num_pos_embs
    coordinator_params['zero_out_pos_embs']= rla_args.gnn_zero_out_pos_embs
    coordinator_params['clip_mode'] = True


    args_dict['arch'] = args.esm_path
    trained_model = model_utils.load_model(path, args_dict['arch'], args.dev)
    tokenizer = EsmTokenizer.from_pretrained(args_dict['arch'])   
    rla_args.blacklist_file = ''
    train_loader, val_loader, train_len, val_len = get_wds_loaders(rla_args, coordinator_params, gpu=None, shuffle_train=False, val_only=False, return_count=False)
    trained_model = trained_model.eval()

    esm_model = EsmForMaskedLM.from_pretrained(args.esm_path)
    base_lm_head = esm_model.lm_head
    lm_up_proj = True
    if lm_up_proj:
        lm_head = ReshapeBaseHead(base_lm_head)
    else:
        lm_head = EsmLMHead()
    lm_head = lm_head.cuda()

    if args.model_type == 'MLP':
        converter = ConverterModel(in_features=[320, 512, 512], out_features=[512, 512, 320], num_layers=3, dropout=0).to(args.dev)
    elif args.model_type == 'transformer':
        converter = ConverterTransformerModel(d_model=320, nhead=4, num_encoder_layers=6, num_decoder_layers=1, dim_feedforward=512, dropout=0.0, batch_first=True, text_feat_type=args.text_feat_type, lm_head=lm_head).to(device=args.dev)
    elif args.model_type == 'encoder':
        converter = ConverterTransformerEncoderModel(d_model=320, nhead=4, num_encoder_layers=6, dim_feedforward=512, dropout=0.0, text_feat_type=args.text_feat_type, lm_head=lm_head).to(device=args.dev)
    loss_fn = torch.nn.MSELoss(reduction='none')

    converter_state = os.path.join(args.run_dir, 'net_best_checkpoint.pt')
    state_dict = torch.load(converter_state, map_location=args.dev)
    converter.load_state_dict(state_dict['state_dict'])

    torch.set_grad_enabled(False)
    converter = converter.eval()

    esm_model = EsmForMaskedLM.from_pretrained(args.esm_path)
    lm_head = esm_model.lm_head
    lm_head = lm_head.to('cuda:0')

    rla_args.batch_size=1
    train_loader, val_loader, train_len, val_len = get_wds_loaders(rla_args, coordinator_params, gpu=None, shuffle_train=False, val_only=False, return_count=False)

    lm_head_new = EsmLMHead(640).to(device='cuda:0')
    # lm_head_base = EsmLMHead(320).to(device='cuda:0')
    lm_head_base = copy.deepcopy(lm_head).cuda()
    lm_head_converter = ReshapeHead(320, copy.deepcopy(lm_head)).cuda()

    base_optimizer = torch.optim.Adam(lm_head_base.parameters(), lr=1e-3, weight_decay=1e-4)
    converter_optimizer = torch.optim.Adam(lm_head_converter.parameters(), lr=1e-3, weight_decay=1e-4)
    new_optimizer = torch.optim.Adam(lm_head_new.parameters(), lr=1e-3, weight_decay=1e-4)

    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, 'tensorboard'))
    training_curves = {"train_loss_base": [], "val_loss_base": [], "train_loss_converter": [], "val_loss_converter": []}
    progress = tqdm(total=100)

    best_val_loss_base = None
    best_val_loss_converter = None

    for epoch in range(100):
        base_train_loss = []
        base_train_nsr = []
        converter_train_loss = []
        converter_train_nsr = []
        new_train_loss = []
        new_train_nsr = []
        base_val_loss = []
        base_val_nsr = []
        converter_val_loss = []
        converter_val_nsr = []
        new_val_loss = []
        new_val_nsr = []
        torch.set_grad_enabled(True)
        lm_head_base.train()
        lm_head_converter.train()
        for i, batch in enumerate(train_loader):
            if len(batch[0]['string_sequence'][0]) < 30 or batch[0]['placeholder_mask'][0].sum().item() < 30:
                continue
            base_loss, base_nsr, converter_loss, converter_nsr, new_loss, new_nsr = run_iter(batch, trained_model, tokenizer, converter, esm, lm_head_base, lm_head_converter, lm_head, base_optimizer, converter_optimizer, new_optimizer, desired_mse=0.0, grad=True)
            base_train_loss.append(base_loss)
            base_train_nsr.append(base_nsr)
            converter_train_loss.append(converter_loss)
            converter_train_nsr.append(converter_nsr)
            new_train_loss.append(new_loss)
            new_train_nsr.append(new_nsr)

        torch.set_grad_enabled(False)
        lm_head_base.eval()
        lm_head_converter.eval()
        for i, batch in enumerate(val_loader):
            if len(batch[0]['string_sequence'][0]) < 30 or batch[0]['placeholder_mask'][0].sum().item() < 30:
                continue
            base_loss, base_nsr, converter_loss, converter_nsr, new_loss, new_nsr = run_iter(batch, trained_model, tokenizer, converter, esm, lm_head_base, lm_head_converter, lm_head, base_optimizer, converter_optimizer, new_optimizer, desired_mse=0.0, grad=False)
            base_val_loss.append(base_loss)
            base_val_nsr.append(base_nsr)
            converter_val_loss.append(converter_loss)
            converter_val_nsr.append(converter_nsr)
            new_val_loss.append(new_loss)
            new_val_nsr.append(new_nsr)
        
        progress.update(1)
        progress.refresh()
        progress.set_description_str(f"val base loss: {np.mean(base_val_loss)} | base nsr: {np.mean(base_val_nsr)} | converter loss: {np.mean(converter_val_loss)} | converter nsr: {np.mean(converter_nsr)} | new loss: {np.mean(new_val_loss)} | new nsr: {np.mean(new_val_nsr)}")
        writer.add_scalar('train loss base', np.mean(base_train_loss), epoch)
        writer.add_scalar('val loss base', np.mean(base_val_loss), epoch)
        writer.add_scalar('train loss converter', np.mean(converter_train_loss), epoch)
        writer.add_scalar('val loss converter', np.mean(converter_val_loss), epoch)
        training_curves["train_loss_base"].append(np.mean(base_train_loss))
        training_curves["val_loss_base"].append(np.mean(base_val_loss))
        training_curves["train_loss_converter"].append(np.mean(converter_train_loss))
        training_curves["val_loss_converter"].append(np.mean(converter_val_loss))
        # Save a state checkpoint
        converter_checkpoint_state = {
            'epoch': epoch,
            'state_dict': lm_head_converter.state_dict(),
            'val_loss': np.mean(converter_val_loss),
            'optimizer_state': converter_optimizer.state_dict(),
            'training_curves': {"train_loss": training_curves["train_loss_converter"], "val_loss": training_curves["val_loss_converter"]}
        }
        torch.save(converter_checkpoint_state, os.path.join(args.out_dir, 'net_last_checkpoint_converter.pt'))

        base_checkpoint_state = {
            'epoch': epoch,
            'state_dict': lm_head_base.state_dict(),
            'val_loss': np.mean(base_val_loss),
            'optimizer_state': base_optimizer.state_dict(),
            'training_curves': {"train_loss": training_curves["train_loss_base"], "val_loss": training_curves["val_loss_base"]}
        }
        torch.save(base_checkpoint_state, os.path.join(args.out_dir, 'net_last_checkpoint_base.pt'))

        if best_val_loss_base is None or np.mean(base_val_loss) < best_val_loss_base:
            torch.save(base_checkpoint_state, os.path.join(args.out_dir, 'net_best_checkpoint_base.pt'))
            best_val_loss_base = np.mean(base_val_loss)
        if best_val_loss_converter is None or np.mean(converter_val_loss) < best_val_loss_converter:
            torch.save(converter_checkpoint_state, os.path.join(args.out_dir, 'net_best_checkpoint_converter.pt'))
            best_val_loss_converter = np.mean(converter_val_loss)

        
    





