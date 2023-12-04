"""Generate structure embedding from a trained COORDinator model.

If indicated, the resulting embeddings will be dumped in :code:`<output_dir>` via
a pickle file :code:`net.out`.

Usage:
    .. code-block::

        python generateStructureEmbedding.py \\
            --dataset <dataset_dir> \\
            --model_dir <trained_model_dir> \\
            [--output_dir <output_dir>] \\
            [--subset <data_subset_file>] \\
            [--dev <device>]

    If :code:`subset` is not provided, the entire dataset :code:`dataset` will
    be evaluated.

See :code:`python generateStructureEmbedding.py --help` for more info.
"""

import argparse
from ast import parse
import json
import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from terminator.data.data import CoordDataset, CoordBatchSampler
from terminator.models.TERMinator import TERMinator
from terminator.utils.model.loop_utils import run_epoch
from terminator.utils.model.loss_fn import construct_loss_fn
from terminator.utils.model.default_hparams import DEFAULT_MODEL_HPARAMS, DEFAULT_TRAIN_HPARAMS

# pylint: disable=unspecified-encoding

def load_params(params_file, hparams_type, default_hparams):
    with open(os.path.join(params_file, hparams_type)) as fp:
        hparams = json.load(fp)
    for key, default_val in default_hparams.items():
        if key not in hparams:
            hparams[key] = default_val
    return hparams


def _to_dev(data_dict, dev):
    """ Push all tensor objects in the dictionary to the given device.

    Args
    ----
    data_dict : dict
        Dictionary of input features to Coordinator
    dev : str
        Device to load tensors onto
    """
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(dev)
        if key == 'gvp_data':
            data_dict['gvp_data'] = [data.to(dev) for data in data_dict['gvp_data']]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate COORDinator embeddings')
    parser.add_argument('--dataset', help='input folder .red.pdb files in proper directory structure', required=True)
    parser.add_argument('--model_dir', help='trained model folder', required=True)
    parser.add_argument('--output_dir', help='where to dump net.out', default=None)
    parser.add_argument('--subset',
                        help=('file specifiying subset of dataset to generate embeddings for. '
                              'if none provided, embeddings will be generated for the whole dataset folder.'))
    parser.add_argument('--dev', help='device to train on', default='cuda:0')
    args = parser.parse_args()

    dev = args.dev
    if torch.cuda.device_count() == 0:
        dev = "cpu"

    if args.subset:
        input_ids = []
        with open(os.path.join(args.subset), 'r') as f:
            for line in f:
                input_ids += [line.strip()]
    else:
        input_ids = None

    model_hparams = load_params(args.model_dir, "model_hparams.json", DEFAULT_MODEL_HPARAMS)
    run_hparams = load_params(args.model_dir, "run_hparams.json", DEFAULT_TRAIN_HPARAMS)

    input_dataset = CoordDataset(args.dataset, pdb_ids=input_ids)
    input_batch_sampler = CoordBatchSampler(input_dataset, batch_size=1, shuffle=False)
    input_dataloader = DataLoader(input_dataset,
                                 batch_sampler=input_batch_sampler,
                                 collate_fn=input_batch_sampler.package)

   

    # backwards compatability
    if "cov_features" not in model_hparams.keys():
        model_hparams["cov_features"] = False
    if "term_use_mpnn" not in model_hparams.keys():
        model_hparams["term_use_mpnn"] = False
    if "matches" not in model_hparams.keys():
        model_hparams["matches"] = "resnet"
    if "struct2seq_linear" not in model_hparams.keys():
        model_hparams['struct2seq_linear'] = False
    if "energies_gvp" not in model_hparams.keys():
        model_hparams['energies_gvp'] = False
    if "num_sing_stats" not in model_hparams.keys():
        model_hparams['num_sing_stats'] = 0
    if "num_pair_stats" not in model_hparams.keys():
        model_hparams['num_pair_stats'] = 0
    if "contact_idx" not in model_hparams.keys():
        model_hparams['contact_idx'] = False
    if "fe_dropout" not in model_hparams.keys():
        model_hparams['fe_dropout'] = 0.1
    if "fe_max_len" not in model_hparams.keys():
        model_hparams['fe_max_len'] = 1000
    if "cie_dropout" not in model_hparams.keys():
        model_hparams['cie_dropout'] = 0.1

    terminator = TERMinator(hparams=model_hparams, device=dev)
    terminator = nn.DataParallel(terminator)

    best_checkpoint_state = torch.load(os.path.join(args.model_dir, 'net_best_checkpoint.pt'), map_location=dev)
    best_checkpoint = best_checkpoint_state['state_dict']
    terminator.module.load_state_dict(best_checkpoint)
    terminator.to(dev)
    terminator.eval()
    torch.set_grad_enabled(False)

    for data in tqdm(input_dataloader):
        data['scatter_idx'] = torch.arange(len(data['seq_lens']))
        _to_dev(data, dev)
        max_seq_len = max(data['seq_lens'].tolist())
        ids = data['ids']
        try:
            etab, E_idx, node_embeddings = terminator(data, max_seq_len)
            etab = etab[0, :, :, :] # Reshape because we know there is only 1 protein per batch
            E_idx = E_idx[0, :, :]
            node_embeddings = node_embeddings[0, :]
            # if output folder specified, save embedding
            if args.output_dir is not None:
                if not os.path.isdir(args.output_dir):
                    os.makedirs(args.output_dir, exist_ok=True)
                with open(os.path.join(args.output_dir, ids[0] + '.etab'), 'wb') as fp:
                    pickle.dump(etab, fp)
                with open(os.path.join(args.output_dir, ids[0] + '.idx'), 'wb') as fp:
                    pickle.dump(E_idx, fp)
                with open(os.path.join(args.output_dir, ids[0] + '.nodes'), 'wb') as fp:
                    pickle.dump(node_embeddings, fp)
        except:
            print(f"Issue with pdb id {ids[0]}. Skipping.")
