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

class JointEmb:
    def __init__(self, h5path, max_len=1024, k_neighbors=30):
        self.fp = h5py.File(h5path, 'r')
        self.max_seq_len = max_len
        self.max_coord_len = self.max_seq_len
        self.k_neighbors = k_neighbors
        
    def __len__(self):
        return self.fp['coord_masks'].shape[0]
    
    def __getitem__(self, idx):
        coord = self.fp['coords'][idx][:self.max_coord_len]
        coord_mask = self.fp['coord_masks'][idx][:self.max_coord_len]
        chain_len = self.fp['chain_lens'][idx][:self.max_coord_len]
        pdb = self.fp['pdbs'][idx].decode('UTF-8')
        res_info = [(res.split(',')[0],int(res.split(',')[1])) for res in self.fp['res_info'][idx].decode('UTF-8').split('_')]
        seq = self.fp['seqs'][idx][:self.max_coord_len]
        seq = seq.decode('UTF-8')
        X, _, _ = _ingraham_featurize([coord])
        X = X[:,:,1,:]
        X = X[:,:len(seq)]
        coord_mask = coord_mask[:len(seq)]
        mask = torch.from_numpy(coord_mask).unsqueeze(0)
        _, _, E_idx = extract_knn(X, mask, eps=1E-6, top_k=self.k_neighbors)
        E_idx = E_idx[0]
        inds_reduce, inds_expand, inds_transpose, inds_duplicate, inds_singles, mask_combs = extract_idxs(E_idx, mask)
        # inds_reduce = inds_reduce.unsqueeze(0)
        # inds_transpose = inds_transpose.unsqueeze(0)
        # inds_duplicate = inds_duplicate.unsqueeze(0)
        # inds_singles = inds_singles.unsqueeze(0)
        # mask_combs = mask_combs.unsqueeze(0)
        inds_convert = (inds_reduce, inds_expand, inds_transpose, inds_duplicate, inds_singles)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.k_neighbors)
        inds_reduce = inds_reduce.to(torch.int64)
        mask_reduced = per_node_to_all_comb(mask_expanded, inds_reduce.unsqueeze(0))
        mask_reduced = torch.multiply(mask_reduced, mask_combs)
        coord_mask = np.pad(coord_mask, ((0, self.max_coord_len - len(seq))), mode='constant')
        return {
            'coords': coord.astype(np.float32), 
            'coords_mask': coord_mask.astype(np.float32), 
            'chain_len': chain_len,
            'pdb': pdb,
            'res_info': res_info,
            'seq': seq, 
            'seq_len': len(seq),
            'inds_convert': inds_convert,
            'mask_reduced': mask_reduced
        }

class COORDDataloader(Dataset):
    """Coordinate dataset that loads coordinate info into a PyTorch Dataset structure"""

    def __init__(self, wd_ds, num_workers=4, batch_size=None):
        super().init(wd_ds, num_workers, batch_size)

    def __getitem__(self, idx):
        """Extract a given item with provided index"""
    
def construct_gnn_inp(b, device, half_precision, burn_in=-1):
    max_seq_len = b['seq_len'].max()
    X = b['coords'][:, :max_seq_len].to(device)
    if half_precision:
        X = X.half()
        
    data_object = {
        'X': X,
        'x_mask': b['coords_mask'][:, :max_seq_len].to(device),
        'chain_idx': b['chain_len'][:, :max_seq_len].to(device),
        'inds_convert': [ind.to(device) for ind in b['inds_convert']],
        'mask_reduced': b['mask_reduced'].to(device)
    }
    
    if burn_in != -1:
        seq_lens = b['seq_len']
        B = len(seq_lens)
        index_vec = torch.arange(max_seq_len).expand(B, max_seq_len)
        burn_mask = (index_vec < seq_lens.unsqueeze(1)-burn_in) & (index_vec >= burn_in)
        loss_mask = data_object['x_mask'].clone()
        loss_mask[~burn_mask] = 0
    else:
        loss_mask = data_object['x_mask'].clone()
        
    return data_object, max_seq_len, loss_mask

def load_params(params_file, default_hparams):
    with open(params_file) as fp:
        hparams = json.load(fp)
    for key, default_val in default_hparams.items():
        if key not in hparams:
            hparams[key] = default_val
    return hparams
    
def get_coordinator_params(params_file):
    return load_params(params_file, DEFAULT_MODEL_HPARAMS)

def adjust_text_features(text_features, inp_dict, tokenizer):
    mask = inp_dict['attention_mask'].clone()
    toks = inp_dict['input_ids']
    eos_token = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    mask[toks == eos_token] = 0
    mask = mask[:, 1:] # ignore class token
    text_features = text_features[:, 1:-1]
    return text_features, mask
    

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