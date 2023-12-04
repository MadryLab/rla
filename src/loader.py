import gzip
import warnings
from pathlib import Path
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser
from Bio import SeqIO
from Bio.SeqUtils import seq1
import webdataset as wds
import sys
import torch
import numpy as np
from terminator.data.data import _ingraham_featurize
from terminator.models.layers.utils import extract_knn, extract_idxs, per_node_to_all_comb
import time
import copy 

RESIDUE_VOCAB = np.array([
            'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 
            'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O'
        ])

def wrap(arr, seq_lens):
    # wrap contiguous array back into array of arrays
    wrapped = []
    curr_idx = 0
    for s in seq_lens:
        out = arr[curr_idx:curr_idx+s]
        if len(out) == 0:
            return wrapped
        else:
            wrapped.append(out)
        curr_idx += s
    return wrapped

def unwrap(wrapped):
    # concatenate list of lists together and return their seq lens
    return np.concatenate(wrapped)

def shuffle_list(order, arr):
    return np.array(arr, dtype=object)[order].tolist()
# ============================
def get_chain_ids(coords, chain_ids):
    base_chain_labels = []
    for i in range(len(chain_ids)):
        chain_len = len(coords[i])
        base_chain_labels.append([chain_ids[i] + str(j) for j in range(chain_len)])
    return base_chain_labels

def get_coord_indices(coords):
    coords_index = []
    curr_idx = 0
    for c in coords:
        coords_index.append(np.arange(len(c)) + curr_idx)
        curr_idx += len(c)
    return coords_index

def get_burn_in_mask(seq_lens, burn_in):
    burn_in_mask = []
    for l in seq_lens:
        chain_mask = np.ones(l).astype(bool)
        if burn_in > 0:
            chain_mask[:burn_in] = False
            chain_mask[-burn_in:] = False
        burn_in_mask.append(chain_mask)
    return burn_in_mask

def get_reduce_masks(coords, k_neighbors):
    t1 = time.time()
    X = torch.from_numpy(coords).unsqueeze(0)[:, :, 1] # 1 x 1 x 4 x 3
    mask = torch.ones(X.shape[1]).unsqueeze(0)
    _, _, E_idx = extract_knn(X, mask, eps=1E-6, top_k=k_neighbors)
    t2 = time.time()
    E_idx = E_idx[0]
    inds_reduce, inds_expand, inds_transpose, inds_duplicate, inds_singles, mask_combs = extract_idxs(E_idx, mask)
    inds_convert = (inds_reduce, inds_expand, inds_transpose, inds_duplicate, inds_singles)
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, k_neighbors)
    inds_reduce = inds_reduce.to(torch.int64)
    mask_reduced = per_node_to_all_comb(mask_expanded, inds_reduce.unsqueeze(0))
    mask_reduced = torch.multiply(mask_reduced, mask_combs)
    t3 = time.time()
    #print("reduce", t3-t2, t2-t1)
    return inds_convert, mask_reduced
# ============================

def link_seqs(seqs, pos_offset, max_seq_len, crop_type='absolute'):
    linker = 'G'*25
    placeholder = ['-'*len(s) for idx, s in enumerate(seqs)]
    
    # what is the length of each sequence
    seq_lens_no_linker = [len(s) for s in seqs]
    seq_lens = [len(s)+len(linker) for s in seqs]
    
    # where does each sequence start
    seq_offsets = [0] + seq_lens
    seq_offsets = np.array(seq_offsets)[:-1]
    
    # add the linker
    seqs = linker.join(seqs)
    placeholder = linker.join(placeholder)
    placeholder_mask = np.array(list(placeholder)) != 'G'
    
    # adjust positional embeddings
    pos_embs = np.arange(len(seqs))
    curr_counter = 0
    for l in seq_lens:
        curr_counter += l
        pos_embs[curr_counter:] += pos_offset
        
    if crop_type == 'positional': # crop based on the positional embedding
        if pos_embs[-1] >= max_seq_len: # need to crop
            cutoff = np.where(pos_embs >= max_seq_len)[0][0]
        else:
            cutoff = len(seqs)
    elif crop_type == 'absolute': # crop based on the actual length
        cutoff = min(len(seqs), max_seq_len)
    cutoff_mask = np.arange(len(seqs)) < cutoff
    
    # max (unwrapped) seq id
    #print(np.array(list(placeholder)).shape, np.where(cutoff_mask[placeholder_mask])[0].shape)
    unwrapped_cutoff = np.where(cutoff_mask[placeholder_mask])[0][-1] + 1
    pos_embs =  pos_embs[:cutoff]
    # add cls and sep token
    pos_embs = pos_embs + 1
    pos_embs = np.concatenate([np.array([0]), pos_embs, pos_embs[[-1]] + 1])
    output = {
        'string_seqs': seqs[:cutoff], # final sequence
        'pos_embs': pos_embs,  # positional embedding to pass to esm
        'placeholder_mask': placeholder_mask[:cutoff], # mask of which things are placeholders
        'unwrapped_cutoff': unwrapped_cutoff 
    }
    return output
    
def _preprocess(batch, shuffle_chains, burn_in):
    # -------------
    # extract from batch and (optionally) perform pre-shuffle of both
    seqs = batch['seqs']
    num_chains = len(seqs)
    coords_dict = {
        'coords': batch['coords'],
        'base_chain_labels': get_chain_ids(batch['coords'], batch['chain_ids']),
        'burn_in_mask': get_burn_in_mask([len(s) for s in seqs], burn_in=burn_in),
    }
    seqs_dict = {
        'seqs': [np.array(list(s)) for s in seqs], # per character 
        'burn_in_mask': get_burn_in_mask([len(s) for s in seqs], burn_in=burn_in)
    }
    
    if shuffle_chains: 
        order = np.random.permutation(num_chains)
        seqs_dict = {k: shuffle_list(order, v) for k, v in seqs_dict.items()}
        coords_dict = {k: shuffle_list(order, v) for k, v in coords_dict.items()}
    return seqs_dict, coords_dict

def _crop_dict(d, seq_lens, max_len):
    return {k: wrap(unwrap(v)[:max_len], seq_lens) for k, v in d.items()}
    
def postprocess(batch, shuffle_chains=True,
                shuffle_coords=True, max_coords_len=6000, 
                max_seq_len=1024, pos_offset=128,
                burn_in=30, k_neighbors=30,
                chain_ends_type='replace',
                crop_type='absolute',
                indiv_mutation=False,
                num_mutations=-1,
                masked_rate=-1,
                masked_mode='MASK',
               ):
    # extract and optionally shuffle both
    if sum([len(u) for u in batch['seqs']]) == 0:
        return None, None
    seqs_dict, coords_dict = _preprocess(batch, shuffle_chains=shuffle_chains, burn_in=burn_in)
    # crop to coords_max_length
    seq_lens = [len(s) for s in seqs_dict['seqs']]
    coords_dict = _crop_dict(coords_dict, seq_lens, max_coords_len)
    coords_dict['coords_index'] = get_coord_indices(coords_dict['coords'])
    seqs_dict = _crop_dict(seqs_dict, seq_lens, max_coords_len)
    seq_lens = [len(s) for s in seqs_dict['seqs']]
    
    # perform shuffle augmentation of just coords
    if shuffle_coords: # perform shuffle augmentation of just coords
        coords_order = np.random.permutation(len(coords_dict['coords']))
        coords_dict = {k: shuffle_list(coords_order, v) for k, v in coords_dict.items()}
    coords_seq_lens = [len(c) for c in coords_dict['coords']]
    
    # perform linkage
    string_seqs = [''.join(s) for s in seqs_dict['seqs']]
    linked_seq_dict = link_seqs(seqs=string_seqs, 
                         pos_offset=pos_offset, 
                         max_seq_len=max_seq_len,
                         crop_type=crop_type
                         )
    
    # debug
    
    # correct coord indices ----------
    unwrapped_cutoff = linked_seq_dict['unwrapped_cutoff'] 
    coords_to_seq = unwrap(coords_dict['coords_index']) # for each coord, what sequence id it refers to
    # mask of the coordinates that actually map to something in ESM
    coords_loss_mask = coords_to_seq < unwrapped_cutoff 
    coords_loss_mask = coords_loss_mask & unwrap(coords_dict['burn_in_mask']) # final loss mask
    
    # get seq indices -----
    seq_loss_mask = unwrap(seqs_dict['burn_in_mask'])
    seq_to_coords = np.ones(len(seq_loss_mask)) * -1
    seq_to_coords[coords_to_seq] = np.arange(len(seq_to_coords))
    seq_to_coords = seq_to_coords[:unwrapped_cutoff]
    seq_loss_mask = seq_loss_mask[:unwrapped_cutoff]
    
    seqs_output = {
        'string_sequence': linked_seq_dict['string_seqs'],
        'pos_embs': linked_seq_dict['pos_embs'],
        'placeholder_mask': linked_seq_dict['placeholder_mask'],
        'seq_loss_mask': seq_loss_mask, # does not count placeholders
        'seq_to_coords': seq_to_coords.astype(int), # does not count placeholders
        'pdb_id': batch['pdb_id'],
    }
    
    coords_output = {
        'coords': unwrap(coords_dict['coords']),
        'res_info': list(unwrap(coords_dict['base_chain_labels'])),
        'seq_lens': coords_seq_lens,
        'coords_loss_mask': coords_loss_mask, # does not count placeholders
        'coords_to_seq': coords_to_seq.astype(int), # does not count placeholders
        'chain_dict': parse_chain_ends(coords_seq_lens, type=chain_ends_type),
#         'chain_lens': batch['chain_lens']
        
    }

    if num_mutations > 0:
        new_seqs, change_coords = _get_mutations(
            orig_seq=seqs_output['string_sequence'],
            pl_mask=seqs_output['placeholder_mask'],
            seq_mask=seqs_output['seq_loss_mask'],
            num_mutations=num_mutations,
        )
        seqs_output['mutation_seqs'] = new_seqs
        seqs_output['coord_to_change'] = change_coords
    if masked_rate > 0:
        llm_masked_sequence, llm_mask = _get_masked(
            orig_seq=seqs_output['string_sequence'],
            pl_mask=seqs_output['placeholder_mask'],
            masked_rate=masked_rate,
            masked_mode=masked_mode,
        )
        seqs_output['llm_masked_sequence'] = llm_masked_sequence
        seqs_output['llm_mask'] = llm_mask
    return seqs_output, coords_output

def _get_masked(orig_seq, pl_mask, masked_rate=0.15, masked_mode='MASK'):
    no_pl_seq_len = pl_mask.sum()
    mutation_mask = np.random.rand(no_pl_seq_len) < masked_rate
    inds = np.arange(len(orig_seq))[pl_mask][mutation_mask]
    orig_seq_arr = [str(u) for u in orig_seq]
    orig_seq_arr = np.array(orig_seq_arr, dtype='<U6')
    if masked_mode == 'MASK':
        orig_seq_arr[inds] = "<mask>"
    elif masked_mode == 'BERT':
        np.random.shuffle(inds)
        N = len(inds)
        orig_seq_arr[inds[:int(N*0.8)]] = "<mask>"
        rand_inds = inds[int(N*0.8):int(N*0.9)]
        orig_seq_arr[rand_inds] = np.random.choice(RESIDUE_VOCAB, size=len(rand_inds))
    elif masked_mode == 'RANDOM':
        np.random.shuffle(inds)
        N = len(inds)
        rand_inds = inds[:int(N*0.5)]
        orig_seq_arr[rand_inds] = np.random.choice(RESIDUE_VOCAB, size=len(rand_inds))
    else:
        assert False
    return ''.join(orig_seq_arr.tolist()), mutation_mask


def _get_mutations(orig_seq, pl_mask, seq_mask, num_mutations=5):
    no_pl_seq_len = pl_mask.sum()
    no_pl_coord_to_change = np.random.choice(np.arange(no_pl_seq_len)[seq_mask]) # coord to change from orig seq
    pl_coord_to_change = np.arange(len(orig_seq))[pl_mask][no_pl_coord_to_change]
    orig_seq_arr = np.array([u for u in orig_seq])
    existing_coord = orig_seq_arr[pl_coord_to_change]
    possible_perturbations = [u for u in RESIDUE_VOCAB if u != existing_coord]
    mutations = np.random.choice(possible_perturbations, num_mutations)
    new_seqs = [orig_seq]
    for i in range(num_mutations):
        arr_copy = copy.deepcopy(orig_seq_arr)
        arr_copy[pl_coord_to_change] = mutations[i]
        new_seqs.append(''.join(arr_copy.tolist()))
    return new_seqs, no_pl_coord_to_change



# Extract chain break and end info
def parse_chain_ends(chain_lens, type='replace'):
    total_len = sum(chain_lens)
    if type == 'replace':
        chain_begin = torch.arange(total_len)
        chain_end = torch.arange(total_len)
    else:
        chain_begin = torch.ones(total_len)
        chain_end = torch.ones(total_len)
    chain_singles = torch.ones(total_len)
    prev_cl = 0
    chain_ids = []
    for i, cl in enumerate(chain_lens):
        if type == 'mask':
            chain_begin[prev_cl] = 0
            chain_end[prev_cl + cl - 1] = 0
        else:
            chain_begin[prev_cl] = min(total_len-1, prev_cl+1)
            chain_end[prev_cl+cl-1] = max(0, prev_cl+cl-2)
        if cl == 1:
            chain_singles[prev_cl] = 0
        prev_cl += cl
        chain_ids += [i]*cl
    return {'begin': chain_begin.long(), 'end': chain_end.long(), 'singles': chain_singles, 'ids': np.array(chain_ids)}

# ================ COLLATION ================

def pad_tensor(tensor_list, min_length=None):
    max_len = max([len(x) for x in tensor_list])
    if min_length is not None:
        max_len = max(max_len, min_length)
    padded_tensors = []
    padded_masks = []
    for t in tensor_list:
        pad_amt = max_len - len(t)
        pad = torch.zeros((pad_amt, *t.shape[1:]))
        padded_tensors.append(torch.cat([t, pad]))
        pad_mask = torch.ones(max_len)
        pad_mask[len(t):] = 0
        padded_masks.append(pad_mask)
    return torch.stack(padded_tensors), torch.stack(padded_masks) == 1

def collate(b, min_length=None):
    if isinstance(b[0], (int, float)):
        return np.array(list(b))
    elif isinstance(b[0], np.ndarray):
        orig_dtype = b[0].dtype
        b, mask = pad_tensor([torch.tensor(u) for u in b], min_length=min_length)
        return (b.numpy().astype(orig_dtype), mask.numpy())
    elif torch.is_tensor(b[0]):
        b, mask = pad_tensor([u for u in b], min_length=min_length)
        return (b, mask)
    elif isinstance(b[0], dict):
        if None in b:
            print(b)
        return {
            k: collate([u[k] for u in b], min_length=min_length)
            for k in b[0].keys()
        }
    else:
        return list(b) 
    return b

def get_custom_collation_fn(min_length=None):
    def custom_collation_fn(samples, combine_tensors=True, combine_scalars=True):
        assert isinstance(samples[0], (list, tuple)), type(samples[0])
        batched = list(zip(*samples))
        result = [collate(b, min_length=min_length) for b in batched]
        return result
    return custom_collation_fn

def partial_custom_collation_fn(samples, combine_tensors=True, combine_scalars=True, min_length=None):
    assert isinstance(samples[0], (list, tuple)), type(samples[0])
    batched = list(zip(*samples))
    result = [collate(b, min_length=min_length) for b in batched]
    return result

def get_filter_fn(min_seq_length, blacklist=None):
    def filter_fn(sample):
        seq_batch, coord_batch = sample
        if seq_batch is None:
            return False
        if coord_batch['coords_loss_mask'].sum() == 0:
            return False
        if len(coord_batch['coords']) <= min_seq_length:
            return False
        if blacklist is not None and seq_batch['pdb_id'].lower() in blacklist:
            return False
        return True
    return filter_fn
