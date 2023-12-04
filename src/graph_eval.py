"""Evaluation code for the graph embeddings learned by CLIP"""

from terminator.models.TERMinator import TERMinator
from clip_main import get_wds_loaders
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
import src.eval_utils as eval_utils
import src.data_utils as data_utils
import src.models_and_optimizers as model_utils
from terminator.models.TERMinator import TERMinator
from transformers import EsmTokenizer
import yaml
from types import SimpleNamespace
import os
import pickle

def graph_eval(config_path, path, coordinator_path, output_dir, use_clip_node_embeddings=True, use_clip_coordinator=True):
    """Transforms the graph embedding output from CLIP into a Potts model for evaluation."""

    # Load parameters, arguments, and models
    with open(config_path, "r") as stream:
        try:
            hparams = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
    args = SimpleNamespace(
        data_root=hparams['training']['data_root'],
        train_wds_path=hparams['data']['train_wds_path'],
        val_wds_path = hparams['data']['val_wds_path'],
        num_workers=hparams['training']['num_workers'],
        batch_size=hparams['training']['batch_size'],
        distributed=0,
    )

    train_loader, val_loader, train_len, val_len = get_wds_loaders(args)
    model = model_utils.load_model(path, 'cuda', coordinator_path)
    dev = next(model.parameters()).device
    tokenizer = EsmTokenizer.from_pretrained(hparams['model']['arch'])
    ckpt = torch.load(path)
    model_building_args = ckpt['run_metadata']['model_building_args']
    coordinator = TERMinator(hparams=model_building_args['terminator_hparams'])
    gnn_state = torch.load(coordinator_path)['state_dict']
    coordinator.load_state_dict(gnn_state)
    projection_dim=320 ## Hard coded for now, but must fix later
    hidden_dim=128 ## Hard coded for now, but must fix later
    embedding_projection = nn.Linear(projection_dim, hidden_dim, bias=False)
    embedding_projection.to(dev)
    coordinator.eval()
    torch.set_grad_enabled(False)
    # Iterate over batches and get Potts model for every batch
    dump = []
    for i_b, batch_ in enumerate(val_loader):
        batch = {
                'coords': batch_[0],
                'coords_mask': batch_[1],
                'chain_len': batch_[2],
                'seq_len': batch_[3],
                'seq': [u['seq'] for u in batch_[4]],
                'ids': [x['pdb'] for x in batch_[5]],
                'res_info': batch_[6],
                'inds_reduce': batch_[7],
                'inds_expand': batch_[8],
                'inds_transpose': batch_[9],
                'inds_duplicate': batch_[10],
                'inds_single': batch_[11],
                'mask_combs': batch[12]
            }
        # print('ids: ',type(batch['ids']),len(batch['ids']),batch['ids'][0:5])
        text_feats, img_feats, mask = data_utils.get_text_and_image_features(batch_, tokenizer, model, hparams)
        if use_clip_node_embeddings:
            img_feats = embedding_projection(img_feats)
        else:
            img_feats = None
        coord_input, max_seq_len = data_utils.construct_gnn_inp(batch, dev, hparams['training']['mixed_precision'] == 1)
        with torch.no_grad():
            with autocast(enabled=True, dtype=torch.float16):
                if use_clip_coordinator:
                    all_etabs, all_E_idx = data_utils.get_potts_model(model, coord_input, max_seq_len, init_node_embeddings=img_feats)
                else:
                    all_etabs, all_E_idx, _ = coordinator(coord_input, max_seq_len=max_seq_len, init_node_embeddings=img_feats)

        for i_prot, (etab, E_idx) in enumerate(zip(all_etabs, all_E_idx)):
            max_res_idx = int(mask[i_prot].sum().item())
            etab = etab[:max_res_idx]
            E_idx = E_idx[:max_res_idx]
            etab = torch.unsqueeze(etab, 0)
            E_idx = torch.unsqueeze(E_idx, 0)
            n_batch, l, n = etab.shape[:3]
            dump.append({
                'loss': 0,
                'out': etab.view(n_batch, l, n, 20, 20).cpu().numpy(),
                'idx': E_idx.cpu().numpy(),
                'ids': [batch['ids'][i_prot]],
                # 'ids': [str(i_b) + str(i_prot)]
                'res_info': [batch['res_info'][i_prot]]
            })

    # Dump to output dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'net.out'), 'wb') as fp:
        pickle.dump(dump, fp)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph eval')
    parser.add_argument('--config_path', help='path to config file', required=True)
    parser.add_argument('--path', help='path to CLIP model', required=True)
    parser.add_argument('--coordinator_path', help='path to coordinator model', required=True)
    parser.add_argument('--output_dir', help='Directory to store output potts models', required=True)
    parser.add_argument('--use_clip_node_embeddings', help='Whether to use node embeddings from clip', default=True)
    parser.add_argument('--use_clip_coordinator', help='Whether to use coordinator updated from clip', default=True)
    args = parser.parse_args()
    graph_eval(args.config_path, args.path, args.coordinator_path, args.output_dir)

