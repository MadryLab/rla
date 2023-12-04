""" Model loop test suite

A collection of tests to ensure that the model loop works.
"""

from collections import namedtuple
import glob
import inspect
import json
import os
import shutil

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from terminator.data.data import (TERMLazyDataset, TERMBatchSampler, TERMDataset, TERMLazyBatchSampler)
from terminator.models.TERMinator import TERMinator
from terminator.utils.model.default_hparams import DEFAULT_MODEL_HPARAMS, DEFAULT_TRAIN_HPARAMS
from terminator.utils.model.loop_utils import run_epoch
from terminator.utils.model.loss_fn import construct_loss_fn
from terminator.utils.model.optim import get_std_opt


# pylint: disable=unspecified-encoding, no-member


torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=1000)
torch.set_printoptions(precision=2)
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(profile="full")

TMP_DIR = "/tmp/test_run_dir"
DATASET = "./data/features/"

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
        with open(hparam_path, 'r') as fp:
            hparams = json.load(fp)
        for key, default_val in default_hparams.items():
            if key not in hparams:
                hparams[key] = default_val

        hparams_path = os.path.join(args.run_dir, output_name)
        if os.path.isfile(hparams_path):
            with open(hparam_path) as fp:
                previous_hparams = json.load(fp)
            if previous_hparams != hparams:
                raise Exception('Given hyperparameters do not agree with previous hyperparameters.')
        else:
            with open(hparams_path, 'w') as fp:
                json.dump(hparams, fp)

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
    train_ids = args.train
    validation_ids = args.validation
    test_ids = args.test

    if args.lazy:
        train_dataset = TERMLazyDataset(args.dataset, pdb_ids=train_ids)
        val_dataset = TERMLazyDataset(args.dataset, pdb_ids=validation_ids)
        test_dataset = TERMLazyDataset(args.dataset, pdb_ids=test_ids)

        train_batch_sampler = TERMLazyBatchSampler(train_dataset,
                                                   batch_size=run_hparams['train_batch_size'],
                                                   shuffle=run_hparams['shuffle'],
                                                   semi_shuffle=run_hparams['semi_shuffle'],
                                                   sort_data=run_hparams['sort_data'],
                                                   term_matches_cutoff=run_hparams['term_matches_cutoff'],
                                                   max_term_res=run_hparams['max_term_res'],
                                                   max_seq_tokens=run_hparams['max_seq_tokens'],
                                                   term_dropout=run_hparams['term_dropout'])
        if 'test_term_matches_cutoff' in run_hparams:
            test_term_matches_cutoff = run_hparams['test_term_matches_cutoff']
        else:
            test_term_matches_cutoff = run_hparams['term_matches_cutoff']
        val_batch_sampler = TERMLazyBatchSampler(val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 term_matches_cutoff=test_term_matches_cutoff)
        test_batch_sampler = TERMLazyBatchSampler(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  term_matches_cutoff=test_term_matches_cutoff)
    else:
        train_dataset = TERMDataset(args.dataset, pdb_ids=train_ids)
        val_dataset = TERMDataset(args.dataset, pdb_ids=validation_ids)
        test_dataset = TERMDataset(args.dataset, pdb_ids=test_ids)

        train_batch_sampler = TERMBatchSampler(train_dataset,
                                               batch_size=run_hparams['train_batch_size'],
                                               shuffle=run_hparams['shuffle'],
                                               semi_shuffle=run_hparams['semi_shuffle'],
                                               sort_data=run_hparams['sort_data'],
                                               max_term_res=run_hparams['max_term_res'],
                                               max_seq_tokens=run_hparams['max_seq_tokens'])
        val_batch_sampler = TERMBatchSampler(val_dataset, batch_size=1, shuffle=False)
        test_batch_sampler = TERMBatchSampler(test_dataset, batch_size=1, shuffle=False)

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


def _setup_model(model_hparams, checkpoint, dev):
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


def _gen_hparam_args(model_hparams, run_hparams):
    """ A hack for creating a fake Namespace """
    ArgTuple = namedtuple("ArgTuple",
                          ["model_hparams",
                           "run_hparams",
                           "run_dir",
                           "dataset",
                           "train",
                           "validation",
                           "test",
                           "lazy"])
    pdb_ids = glob.glob("./data/features/*/*.features")
    pdb_ids = [os.path.splitext(os.path.basename(path))[0] for path in pdb_ids]
    args = ArgTuple(model_hparams=model_hparams,
                    run_hparams=run_hparams,
                    run_dir=TMP_DIR,
                    dataset=DATASET,
                    train=pdb_ids,
                    validation=pdb_ids,
                    test=pdb_ids,
                    lazy=True)
    return args


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    """ Fixture to execute asserts before and after a test is run """
    # a slight hack to bring us to the test file directory
    test_file_path = os.path.abspath(inspect.getfile(test_coordinator_setup))
    test_file_dir = os.path.dirname(test_file_path)
    os.chdir(test_file_dir)
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.mkdir(TMP_DIR)
    yield # this is where the testing happens
    shutil.rmtree(TMP_DIR)


def test_coordinator_setup():
    """ Test creating a COORDinator model """
    args = _gen_hparam_args("./data/hparams/coordinator_model_hparams.json",
                            "./data/hparams/coordinator_run_hparams.json")
    model_hparams, run_hparams = _setup_hparams(args)
    coordinator = _setup_model(model_hparams, None, dev="cpu")
    return coordinator, model_hparams, run_hparams


def test_terminator_setup():
    """ Test creating a TERMinator model """
    args = _gen_hparam_args("./data/hparams/terminator_model_hparams.json",
                            "./data/hparams/terminator_run_hparams.json")
    model_hparams, run_hparams = _setup_hparams(args)
    terminator = _setup_model(model_hparams, None, dev="cpu")
    return terminator, model_hparams, run_hparams


def test_load_data():
    """ Test creating the dataloaders """
    args = _gen_hparam_args("./data/hparams/coordinator_model_hparams.json",
                            "./data/hparams/coordinator_run_hparams.json")
    _, run_hparams = _setup_hparams(args)
    return _setup_dataloaders(args, run_hparams)


def test_train_terminator():
    """ Test a training epoch of TERMinator """
    args = _gen_hparam_args("./data/hparams/terminator_model_hparams.json",
                            "./data/hparams/terminator_run_hparams.json")
    model_hparams, run_hparams = _setup_hparams(args)
    terminator, _ = _setup_model(model_hparams, None, dev="cpu")
    train_dl, _, _ = _setup_dataloaders(args, run_hparams)
    loss_fn = construct_loss_fn(run_hparams)
    optimizer = get_std_opt(terminator.parameters(),
                            d_model=model_hparams['energies_hidden_dim'],
                            regularization=run_hparams['regularization'],
                            state=None,
                            finetune=False,
                            finetune_lr=run_hparams["finetune_lr"])
    epoch_loss, _, _ = run_epoch(terminator,
                                 train_dl,
                                 loss_fn,
                                 optimizer=optimizer,
                                 grad=True,
                                 dev="cpu",
                                 finetune=False,
                                 isDataParallel=False)


def test_val_terminator():
    """ Test a validation epoch of terminator """
    args = _gen_hparam_args("./data/hparams/terminator_model_hparams.json",
                            "./data/hparams/terminator_run_hparams.json")
    model_hparams, run_hparams = _setup_hparams(args)
    terminator, _ = _setup_model(model_hparams, None, dev="cpu")
    _, val_dl, _ = _setup_dataloaders(args, run_hparams)
    loss_fn = construct_loss_fn(run_hparams)
    epoch_loss, _, _ = run_epoch(terminator,
                                 val_dl,
                                 loss_fn,
                                 dev="cpu",
                                 finetune=False,
                                 isDataParallel=False)


def test_inf_terminator():
    """ Test a test epoch of terminator """
    args = _gen_hparam_args("./data/hparams/terminator_model_hparams.json",
                            "./data/hparams/terminator_run_hparams.json")
    model_hparams, run_hparams = _setup_hparams(args)
    terminator, _ = _setup_model(model_hparams, None, dev="cpu")
    _, _, test_dl = _setup_dataloaders(args, run_hparams)
    loss_fn = construct_loss_fn(run_hparams)
    run_epoch(terminator,
              test_dl,
              loss_fn,
              test=True,
              dev="cpu",
              finetune=False,
              isDataParallel=False)


def test_train_coordinator():
    """ Test a training epoch of TERMinator """
    args = _gen_hparam_args("./data/hparams/coordinator_model_hparams.json",
                            "./data/hparams/coordinator_run_hparams.json")
    model_hparams, run_hparams = _setup_hparams(args)
    coordinator, _ = _setup_model(model_hparams, None, dev="cpu")
    train_dl, _, _ = _setup_dataloaders(args, run_hparams)
    loss_fn = construct_loss_fn(run_hparams)
    optimizer = get_std_opt(coordinator.parameters(),
                            d_model=model_hparams['energies_hidden_dim'],
                            regularization=run_hparams['regularization'],
                            state=None,
                            finetune=False,
                            finetune_lr=run_hparams["finetune_lr"])
    epoch_loss, _, _ = run_epoch(coordinator,
                                 train_dl,
                                 loss_fn,
                                 optimizer=optimizer,
                                 grad=True,
                                 dev="cpu",
                                 finetune=False,
                                 isDataParallel=False)


def test_val_coordinator():
    """ Test a validation epoch of coordinator """
    args = _gen_hparam_args("./data/hparams/coordinator_model_hparams.json",
                            "./data/hparams/coordinator_run_hparams.json")
    model_hparams, run_hparams = _setup_hparams(args)
    coordinator, _ = _setup_model(model_hparams, None, dev="cpu")
    _, val_dl, _ = _setup_dataloaders(args, run_hparams)
    loss_fn = construct_loss_fn(run_hparams)
    epoch_loss, _, _ = run_epoch(coordinator,
                                 val_dl,
                                 loss_fn,
                                 dev="cpu",
                                 finetune=False,
                                 isDataParallel=False)


def test_inf_coordinator():
    """ Test a test epoch of coordinator """
    args = _gen_hparam_args("./data/hparams/coordinator_model_hparams.json",
                            "./data/hparams/coordinator_run_hparams.json")
    model_hparams, run_hparams = _setup_hparams(args)
    coordinator, _ = _setup_model(model_hparams, None, dev="cpu")
    _, _, test_dl = _setup_dataloaders(args, run_hparams)
    loss_fn = construct_loss_fn(run_hparams)
    run_epoch(coordinator,
              test_dl,
              loss_fn,
              test=True,
              dev="cpu",
              finetune=False,
              isDataParallel=False)