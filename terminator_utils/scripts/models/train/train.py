"""Train TERMinator model.

Usage:
    .. code-block::

        python train.py \\
            --dataset <dataset_dir> \\
            --model_hparams <model_hparams_file_path> \\
            --run_hparams <run_hparams_file_path> \\
            --run_dir <run_dir> \\
            [--train <train_split_file>] \\
            [--validation <val_split_file>] \\
            [--test <test_split_file>] \\
            [--out_dir <out_dir>] \\
            [--dev <device>] \\
            [--epochs <num_epochs>]
            [--lazy]

    If :code:`--out_dir <out_dir>` is not set, :code:`net.out` will be dumped
    into :code:`<run_dir>`.

    For any of the split files, if the option is not provided, :code:`train.py` will
    look for them within :code:`<dataset_dir>`.

See :code:`python train.py --help` for more info.
"""

import argparse
import copy
import json
import os
import pickle
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../../..'))
sys.path.insert(0, '/data1/groups/keating_madry')
from coordinator_webdataset import webdataset as wds
# from terminator.data.data import (TERMLazyDataset, TERMBatchSampler, TERMDataset, TERMLazyBatchSampler)
from terminator.data.data import (CoordLazyDataset, CoordBatchSampler, CoordDataset, CoordLazyBatchSampler)
from terminator.models.TERMinator import TERMinator
from terminator.utils.model.loop_utils import run_epoch
from terminator.utils.model.loss_fn import construct_loss_fn

# for autosummary import purposes
# pylint: disable=wrong-import-order,wrong-import-position
# sys.path.insert(0, os.path.dirname(__file__))
from terminator.utils.model.default_hparams import DEFAULT_MODEL_HPARAMS, DEFAULT_TRAIN_HPARAMS
from terminator.utils.model.optim import get_std_opt

# pylint: disable=unspecified-encoding


torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=1000)
torch.set_printoptions(precision=2)
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(profile="full")


def _setup_hparams(args):
    """ Setup the hparams dictionary using defaults and return it

    Args
    ----
    args : argparse.Namespace
        Parsed arguments

    Returns
    -------
    model_hparams : dict
        Fully configured model hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)
    run_hparams : dict
        Fully configured training run hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)
    """
    def _load_hparams(hparam_path, default_hparams, output_name):
        # load hparams
        hparams = json.load(open(hparam_path, 'r'))
        for key, default_val in default_hparams.items():
            if key not in hparams:
                hparams[key] = default_val

        hparams_path = os.path.join(args.run_dir, output_name)
        if os.path.isfile(hparams_path):
            previous_hparams = json.load(open(hparams_path, 'r'))
            if previous_hparams != hparams:
                raise Exception('Given hyperparameters do not agree with previous hyperparameters.')
        else:
            json.dump(hparams, open(hparams_path, 'w'))

        return hparams

    model_hparams = _load_hparams(args.model_hparams, DEFAULT_MODEL_HPARAMS, 'model_hparams.json')
    run_hparams = _load_hparams(args.run_hparams, DEFAULT_TRAIN_HPARAMS, 'run_hparams.json')

    return model_hparams, run_hparams


def _setup_dataloaders(args, run_hparams):
    """ Setup dataloaders needed for training

    Args
    ----
    args : argparse.Namespace
        Parsed arguments
    run_hparams : dict
        Fully configured hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)

    Returns
    -------
    train_dataloader, val_dataloader, test_dataloader : torch.utils.data.DataLoader
        DataLoaders for the train, validation, and test datasets
    """
    kwargs = {}
    kwargs['num_workers'] = 16

    # set up dataloaders
    train_ids = []
    with open(args.train, 'r') as f:
        for line in f:
            train_ids += [line[:-1]]
    validation_ids = []
    with open(args.validation, 'r') as f:
        for line in f:
            validation_ids += [line[:-1]]
    test_ids = []
    with open(args.test, 'r') as f:
        for line in f:
            test_ids += [line[:-1]]
    if args.lazy:
        train_dataset = CoordLazyDataset(args.dataset, pdb_ids=train_ids)
        val_dataset = CoordLazyDataset(args.dataset, pdb_ids=validation_ids)
        test_dataset = CoordLazyDataset(args.dataset, pdb_ids=test_ids)

        train_batch_sampler = CoordLazyBatchSampler(train_dataset,
                                                   batch_size=run_hparams['train_batch_size'],
                                                   shuffle=run_hparams['shuffle'],
                                                   semi_shuffle=run_hparams['semi_shuffle'],
                                                   sort_data=run_hparams['sort_data'],
                                                #    term_matches_cutoff=run_hparams['term_matches_cutoff'],
                                                #    max_term_res=run_hparams['max_term_res'],
                                                   max_seq_tokens=run_hparams['max_seq_tokens'])
                                                #    term_dropout=run_hparams['term_dropout'])
        if 'test_term_matches_cutoff' in run_hparams:
            test_term_matches_cutoff = run_hparams['test_term_matches_cutoff']
        else:
            test_term_matches_cutoff = run_hparams['term_matches_cutoff']
        val_batch_sampler = CoordLazyBatchSampler(val_dataset,
                                                 batch_size=1,
                                                 shuffle=False)
                                                #  term_matches_cutoff=test_term_matches_cutoff)
        test_batch_sampler = CoordLazyBatchSampler(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False)
                                                #   term_matches_cutoff=test_term_matches_cutoff)
    else:
        train_dataset = CoordDataset(args.dataset, pdb_ids=train_ids)
        val_dataset = CoordDataset(args.dataset, pdb_ids=validation_ids)
        test_dataset = CoordDataset(args.dataset, pdb_ids=test_ids)

        train_batch_sampler = CoordBatchSampler(train_dataset,
                                               batch_size=run_hparams['train_batch_size'],
                                               shuffle=run_hparams['shuffle'],
                                               semi_shuffle=run_hparams['semi_shuffle'],
                                               sort_data=run_hparams['sort_data'],
                                            #    max_term_res=run_hparams['max_term_res'],
                                               max_seq_tokens=run_hparams['max_seq_tokens'])
        val_batch_sampler = CoordBatchSampler(val_dataset, batch_size=1, shuffle=False)
        test_batch_sampler = CoordBatchSampler(test_dataset, batch_size=1, shuffle=False)

    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_batch_sampler,
                                  collate_fn=train_batch_sampler.package,
                                  pin_memory=True,
                                  **kwargs)
    val_dataloader = DataLoader(val_dataset,
                                batch_sampler=val_batch_sampler,
                                collate_fn=val_batch_sampler.package,
                                pin_memory=True,
                                **kwargs)
    test_dataloader = DataLoader(test_dataset,
                                 batch_sampler=test_batch_sampler,
                                 collate_fn=test_batch_sampler.package,
                                 **kwargs)

    return train_dataloader, val_dataloader, test_dataloader


def _setup_dataloaders_wds(args):
    print("=> using WebDataset loader")
    cols = ['coords.pyd', 'coords_mask.pyd', 'chain_len.pyd', 'seq_len.pyd', 'seq.pyd', 'pdb.pyd', 
        'inds_convert.pyd', 'mask_reduced.pyd']
    train_dataset = wds.WebDataset(os.path.join(args.wds_root,"train_wds_multichain.tar"), resampled=True).decode().to_tuple(*cols)
    val_dataset = wds.WebDataset(os.path.join(args.wds_root,"val_wds_multichain.tar"), resampled=True).decode().to_tuple(*cols)
    test_dataset = wds.WebDataset(os.path.join(args.wds_root,"test_wds_multichain.tar"), resampled=True).decode().to_tuple(*cols)
    train_dataset = train_dataset.shuffle(100)
    train_dataset = train_dataset.batched(None)
    val_dataset = val_dataset.batched(1)
    test_dataset = test_dataset.batched(1)
    # WebLoader is just the regular DataLoader with the same convenience methods
    # that WebDataset has.
    train_loader = wds.WebLoader(train_dataset, batch_size=None, shuffle=False, num_workers=args.workers)
    val_loader = wds.WebLoader(val_dataset, batch_size=None, shuffle=False, num_workers=1)
    test_loader = wds.WebLoader(test_dataset, batch_size=None, shuffle=False, num_workers=1)
    return train_loader, val_loader, test_loader



def _load_checkpoint(run_dir, dev, finetune=False):
    """ If a training checkpoint exists, load the checkpoint. Otherwise, setup checkpointing initial values.

    Args
    ----
    run_dir : str
        Path to directory containing the training run checkpoint, as well the tensorboard output.

    Returns
    -------
    dict
        Dictionary containing
        - "best_checkpoint_state": the best checkpoint state during the run
        - "last_checkpoint_state": the most recent checkpoint state during the run
        - "best_checkpoint": the best model parameter set during the run
        - "best_validation": the best validation loss during the run
        - "last_optim_state": the most recent state of the optimizer
        - "start_epoch": what epoch to resume training from
        - "writer": SummaryWriter for tensorboard
        - "training_curves": pairs of (train_loss, val_loss) representing the training and validation curves
    """

    if os.path.isfile(os.path.join(run_dir, 'net_best_checkpoint.pt')):
        best_checkpoint_state = torch.load(os.path.join(run_dir, 'net_best_checkpoint.pt'), map_location=torch.device(dev))
        last_checkpoint_state = torch.load(os.path.join(run_dir, 'net_last_checkpoint.pt'), map_location=torch.device(dev))
        best_checkpoint = best_checkpoint_state['state_dict']
        best_validation = best_checkpoint_state['val_loss']
        last_optim_state = last_checkpoint_state["optimizer_state"]
        start_epoch = last_checkpoint_state['epoch'] + 1
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'), purge_step=start_epoch + 1)
        training_curves = last_checkpoint_state["training_curves"]
    else:
        best_checkpoint_state, last_checkpoint_state = None, None
        best_checkpoint = None
        best_validation = 10e8
        last_optim_state = None
        start_epoch = 0
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'))
        training_curves = {"train_loss": [], "val_loss": []}
        if finetune: # load existing model for finetuning
            best_checkpoint_state = torch.load(os.path.join(run_dir, 'net_original.pt'), map_location=torch.device(dev))
            best_checkpoint = best_checkpoint_state['state_dict']

    return {"best_checkpoint_state": best_checkpoint_state,
            "last_checkpoint_state": last_checkpoint_state,
            "best_checkpoint": best_checkpoint,
            "best_validation": best_validation,
            "last_optim_state": last_optim_state,
            "start_epoch": start_epoch,
            "writer": writer,
            "training_curves": training_curves}


def _setup_model(model_hparams, run_hparams, checkpoint, dev):
    """ Setup a TERMinator model using hparams, a checkpoint if provided, and a computation device.

    Args
    ----
    model_hparams : dict
        Fully configured model hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)
    run_hparams : dict
        Fully configured training run hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)
    checkpoint : OrderedDict or None
        Model parameters
    dev : str
        Computation device to use

    Returns
    -------
    terminator : TERMinator or nn.DataParallel(TERMinator)
        Potentially parallelized TERMinator to use for training
    terminator_module : TERMinator
        Inner TERMinator, unparallelized
    """
    terminator = TERMinator(hparams=model_hparams, device=dev)
    if checkpoint is not None:
        terminator.load_state_dict(checkpoint)
    print(terminator)
    print("terminator hparams", terminator.hparams)

    if torch.cuda.device_count() > 1 and dev != "cpu":
        terminator = nn.DataParallel(terminator)
        terminator_module = terminator.module
    else:
        terminator_module = terminator
    terminator.to(dev)

    return terminator, terminator_module


def main(args):
    """ Train TERMinator """
    dev = args.dev
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    # setup dataloaders
    model_hparams, run_hparams = _setup_hparams(args)
    if not args.wds_dataloader:
        train_dataloader, val_dataloader, test_dataloader = _setup_dataloaders(args, run_hparams)
    else:
        train_dataloader, val_dataloader, test_dataloader = _setup_dataloaders_wds(args)
    # load checkpoint
    checkpoint_dict = _load_checkpoint(run_dir, dev, run_hparams['finetune'])
    best_validation = checkpoint_dict["best_validation"]
    best_checkpoint = checkpoint_dict["best_checkpoint"]
    start_epoch = checkpoint_dict["start_epoch"]
    last_optim_state = checkpoint_dict["last_optim_state"]
    writer = checkpoint_dict["writer"]
    training_curves = checkpoint_dict["training_curves"]

    isDataParallel = True if torch.cuda.device_count() > 1 and dev != "cpu" else False
    finetune = run_hparams["finetune"]

    # construct terminator, loss fn, and optimizer
    terminator, terminator_module = _setup_model(model_hparams, run_hparams, best_checkpoint, dev)
    loss_fn = construct_loss_fn(run_hparams)
    optimizer = get_std_opt(terminator.parameters(),
                            d_model=model_hparams['energies_hidden_dim'],
                            regularization=run_hparams['regularization'],
                            state=last_optim_state,
                            finetune=finetune,
                            finetune_lr=run_hparams["finetune_lr"])

    try:
        for epoch in range(start_epoch, args.epochs):
            print('epoch', epoch)

            epoch_loss, epoch_ld, _ = run_epoch(terminator, train_dataloader, loss_fn, optimizer=optimizer, grad=True, dev=dev, finetune=finetune, isDataParallel=isDataParallel, wds_dataloader=args.wds_dataloader)
            print('epoch loss', epoch_loss, 'epoch_ld', epoch_ld)
            writer.add_scalar('training loss', epoch_loss, epoch)

            # validate
            val_loss, val_ld, _ = run_epoch(terminator, val_dataloader, loss_fn, grad=False, dev=dev, wds_dataloader=args.wds_dataloader)
            print('val loss', val_loss, 'val ld', val_ld)
            writer.add_scalar('val loss', val_loss, epoch)

            training_curves["train_loss"].append((epoch_loss, epoch_ld))
            training_curves["val_loss"].append((val_loss, val_ld))

            # comp = (val_ld['sortcery_loss']['loss'] < best_validation) if finetune else (val_loss < best_validation)
            comp = val_loss < best_validation

            # save a state checkpoint
            checkpoint_state = {
                'epoch': epoch,
                'state_dict': terminator_module.state_dict(),
                'best_model': comp, # (val_loss < best_validation)
                'val_loss': best_validation,
                'optimizer_state': optimizer.state_dict(),
                'training_curves': training_curves
            }
            torch.save(checkpoint_state, os.path.join(run_dir, 'net_last_checkpoint.pt'))
            if comp: # if (val_loss < best_validation)
                # if finetune:
                #     best_validation = val_ld['sortcery_loss']['loss']
                # else:
                best_validation = val_loss
                best_checkpoint = copy.deepcopy(terminator_module.state_dict())
                torch.save(checkpoint_state, os.path.join(run_dir, 'net_best_checkpoint.pt'))

    except KeyboardInterrupt:
        pass

    # save model params
    print(training_curves)
    torch.save(terminator_module.state_dict(), os.path.join(run_dir, 'net_last.pt'))
    torch.save(best_checkpoint, os.path.join(run_dir, 'net_best.pt'))

    # test
    terminator_module.load_state_dict(best_checkpoint)
    test_loss, test_ld, dump = run_epoch(terminator, test_dataloader, loss_fn, grad=False, test=True, dev=dev)
    print(f"test loss {test_loss} test loss dict {test_ld}")
    # dump outputs
    if args.out_dir:
        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)
        net_out_path = os.path.join(args.out_dir, "net.out")
    else:
        net_out_path = os.path.join(run_dir, "net.out")
    # save etab outputs for dTERMen runs
    with open(net_out_path, 'wb') as fp:
        pickle.dump(dump, fp)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train TERMinator!')
    parser.add_argument('--dataset', help='input folder .red.pdb files in proper directory structure.', required=True)
    parser.add_argument('--model_hparams', help='file path for model hparams', required=True)
    parser.add_argument('--run_hparams', help='file path for run hparams', required=True)
    parser.add_argument('--run_dir', help='path to place folder to store model files', required=True)
    parser.add_argument('--train', help='file with training dataset split')
    parser.add_argument('--validation', help='file with validation dataset split')
    parser.add_argument('--test', help='file with test dataset split')
    parser.add_argument('--out_dir',
                        help='path to place test set eval results (e.g. net.out). If not set, default to --run_dir')
    parser.add_argument('--dev', help='device to train on', default='cuda:0')
    parser.add_argument('--epochs', help='number of epochs to train for', default=100, type=int)
    parser.add_argument('--lazy', help="use lazy data loading", action='store_true')
    parser.add_argument("--wds_dataloader", help="Whether to use wds dataloader", type=bool, default=False)

    parsed_args = parser.parse_args()

    # by default, if no splits are provided, read the splits from the dataset folder
    if parsed_args.train is None:
        parsed_args.train = os.path.join(parsed_args.dataset, 'train.in')
    if parsed_args.validation is None:
        parsed_args.validation = os.path.join(parsed_args.dataset, 'validation.in')
    if parsed_args.test is None:
        parsed_args.test = os.path.join(parsed_args.dataset, 'test.in')

    main(parsed_args)
