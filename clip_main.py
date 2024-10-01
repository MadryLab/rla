# from wandb import Config
#python clip_main.py --config dataset_configs/rn50_clip.yaml --training.exp_name temp
import copy
import os
import uuid

import torch
import torch as ch
import torch.nn as nn
from fastargs import Param, Section
from fastargs.validation import And, OneOf
import numpy as np
import src.config_parse_utils as config_parse_utils
from src.eval_utils import evaluate_model
from src.trainer import LightWeightTrainer
from src.models_and_optimizers import create_clip_model, load_model
import src.dist_utils as dist_utils
import src.data_utils as data_utils
from transformers import EsmTokenizer
import src.loader as loaders_utils
import sys
import webdataset as wds
import tqdm
import tensorflow as tf
import os
import logging
from functools import partial
import src.loader as loader_utils
import src.zipdataset as zipdataset_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

Section("training", "training arguments").params(
    num_workers=Param(int, "number of workers", default=10),
    batch_size=Param(int, "batch size", default=512),
    exp_name=Param(str, "experiment name", default=""),
    epochs=Param(int, "max epochs", default=60),
    lr=Param(float, "learning rate", default=0.1),
    weight_decay=Param(float, "weight decay", default=1e-4),
    momentum=Param(float, "SGD momentum", default=0.9),
    lr_peak_epoch=Param(int, "lr_peak_epoch for cyclic lr schedular", default=5),
    label_smoothing=Param(float, "label smoothing", default=0.0),
    disable_logging=Param(int, "disable logging", default=0),
    data_root=Param(str, "data root dir", default="/mnt/cfs/projects/proteins/datasets/"),
    eval_epochs=Param(int, "Evaluate every n epochs.", default=2),
    out_dir=Param(str, "output directory", default="runs/"),
    only_non_bn_weight_decay=Param(bool, "only apply WD to non BN params", default=False),
    opt=Param(str, 'type of optimizer', default='SGD'),
    lr_scheduler=Param(str, 'type of lr_scheduler', default='cyclic'),
    mixed_precision=Param(int, 'whether to use mixed precision', default=0),
    max_seq_len=Param(int, 'max sequence length', default=1024),
    self_supervised=Param(int, 'use self sup loss', default=1),
    burn_in=Param(int, 'leading an trailing proteins to ignore', default=-1),
    max_coord_len=Param(int, 'max coords length', default=2000),
    freeze_llm=Param(int, 'whether to freeze language model', default=0),
    freeze_text_proj=Param(int, 'whether to freeze language model projection', default=0),
    use_text_proj=Param(int, 'whether to use text projection layer', default=1),
    projection_dim=Param(int, 'dimension of projection layer', default=320),
    finetune_from=Param(str, 'finetune from a checkpoint', default=''),
    # mutation parameters
    num_mutations=Param(int, 'how many mutations to add for indiv mutation loss', default=-1),
    mutation_fwd_method=Param(str, 'mutation loss fwd method', default='all'),
    # lm parameters
    masked_rate=Param(float, "masking rate", default=-1),
    masked_mode=Param(str, "type of masking", default='MASK'), # "MASK" or "BERT"
    lm_only_text=Param(int, "whether to only supervise text", default=1),
    # weighting
    lm_weight=Param(float, "multiplier for language loss", default=1),
    resid_weight=Param(float, "multiplier for residual loss", default=1),
)

Section("clip_batching", "batching for CLIP arguments").params(
    zip_enabled=Param(int, "whether to enable special CLIP batching", default=0),
    zip_train_format_string=Param(str, "format string for train", default=''),
    cath_info_dict=Param(str, "where is the information on cath augmented", default=''),
    zip_num_steps_per_epoch=Param(int, "how many steps to perform per epoch"),
)

Section("model", "model architecture arguments").params(
    arch=Param(str, "architecture to train", default="RN50"),
    coordinator_hparams=Param(str, "path to coordinator hparams", default=''),
    gnn_checkpoint=Param(str, "path to gnn checkpoint", default=''),
    gnn_num_pos_embs=Param(int, "for gnn, number of positional embeddings", default=16),
    gnn_zero_out_pos_embs=Param(bool, "for gnn, whether to zero out the pos embs", default=False),
    language_head=Param(int, "whether to add a language model head", default=False),
    language_head_type=Param(str, default="MLP"), #MLP or LINEAR
)

Section("data", "data arguments").params(
    train_wds_path=Param(str, "path of train webdatset", default="wds/train_wds_multichain.tar"),
    val_wds_path=Param(str, "path of val webdatset", default="wds/val_wds_multichain.tar"),
    sharded=Param(int, "whether dataset is sharded", default=0),
    blacklist_file=Param(str, "list of blacklisted pdbs", default=''),
)

Section('distributed', 'distributed training options').params(
    distributed=Param(int, 'whether to run in dist mode', default=0),
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355'),
    dist_train_len=Param(int, 'train num examples (needed for distributed)', default=-1),
    dist_val_len=Param(int, 'val num examples (needed for distributed)', default=-1),
)

def get_postprocess_args(args, coordinator_params, shuffle):
    return {
        'max_coords_len': args.max_coord_len,
        'shuffle_coords': shuffle,
        'max_seq_len': args.max_seq_len,
        'pos_offset': 128,
        'burn_in': args.burn_in,
        'k_neighbors': coordinator_params['k_neighbors'],
        'crop_type': 'absolute',
        'shuffle_chains': shuffle,
        'num_mutations': args.num_mutations,
        'masked_rate': args.masked_rate,
        'masked_mode': args.masked_mode,
    }

def _data_aug(sample, postprocess_args):
    return loaders_utils.postprocess(sample[0], **postprocess_args)

def _filter_fn(sample, k_neighbors, blacklist):
    return loader_utils.get_filter_fn(k_neighbors, blacklist)(sample)


def _collate(samples, combine_tensors=True, combine_scalars=True, k_neighbors=30):
    return loader_utils.get_custom_collation_fn(k_neighbors)(
        samples=samples, combine_tensors=combine_tensors, combine_scalars=combine_scalars)

def get_blacklist(args):
    if args.blacklist_file == '':
        return None
    else:
        with open(args.blacklist_file, 'r') as file:
            lines = [line.rstrip().lower() for line in file]
        return lines


def create_distributed_wds_loader(args, coordinator_params, path, shuffle, count, blacklist=None):
    cols = ['inp.pyd']
    resampled = args.distributed == 1
    wd_ds = wds.WebDataset(path, resampled=resampled, shardshuffle=shuffle).decode().to_tuple(*cols)
    postprocess_args = get_postprocess_args(args, coordinator_params, shuffle) 
    ds = wd_ds.map(partial(_data_aug, postprocess_args=postprocess_args))
    if coordinator_params['energies_style'] != 'graphformer':
        min_length = coordinator_params['k_neighbors']
    else:
        min_length = 30

    ds = ds.select(partial(_filter_fn, k_neighbors=coordinator_params['k_neighbors'], blacklist=blacklist))
        
    count = int((count / args.batch_size) / args.world_size)
    if shuffle:
        ds = ds.shuffle(100)
    assert count > 0
    #custom_collation_fn = loader_utils.get_custom_collation_fn(coordinator_params['k_neighbors'])
    custom_collation_fn = partial(_collate, k_neighbors=coordinator_params['k_neighbors'])
    batched_ds = ds.batched(args.batch_size, collation_fn=custom_collation_fn) #.with_length(count)
    dl = torch.utils.data.DataLoader(batched_ds, num_workers=args.num_workers, batch_size=None)
    return dl, count

def create_ziploader(args, coordinator_params, path, shuffle, return_count=True, dist_args=None, is_val=False):
    print("using zip loader")
    rng = np.random.default_rng(12345)
    if not is_val:
        wd_ds = zipdataset_utils.get_clip_webdataset(
            path, args.zip_train_format_string, args.cath_info_dict, 
            args.zip_num_steps_per_epoch, args.batch_size,
            rng=rng, dist_args=dist_args,
        )
    else:
        cols = ['inp.pyd']
        wd_ds = wds.WebDataset(path, shardshuffle=False).decode().to_tuple(*cols)
    postprocess_args = get_postprocess_args(args, coordinator_params, shuffle) 
    ds = wd_ds.map(partial(_data_aug, postprocess_args=postprocess_args))
    if dist_args == None:
        custom_collation_fn = loader_utils.get_custom_collation_fn(coordinator_params['k_neighbors'])
    else:
        custom_collation_fn = partial(loader_utils.partial_custom_collation_fn, min_length=coordinator_params['k_neighbors'])
    
    batched_ds = ds.batched(args.batch_size, collation_fn=custom_collation_fn)
    dl = torch.utils.data.DataLoader(batched_ds, num_workers=args.num_workers, batch_size=None)
    if dist_args is None:
        count = 0
        if return_count:
            for u in tqdm.tqdm(dl):
                count += 1
    else:
        count = args.zip_num_steps_per_epoch
    return dl, count

def create_simple_wds_loader(args, coordinator_params, path, shuffle, return_count=True, blacklist=None):
    cols = ['inp.pyd']
    wd_ds = wds.WebDataset(path, shardshuffle=shuffle).decode().to_tuple(*cols)
    postprocess_args = get_postprocess_args(args, coordinator_params, shuffle) 
    print(postprocess_args)
    def data_aug(sample):
        return loaders_utils.postprocess(sample[0], **postprocess_args)
    ds = wd_ds.map(data_aug)

    if coordinator_params['energies_style'] != 'graphformer':
        min_length = coordinator_params['k_neighbors']
    else:
        min_length = 30
    filter_fn = loader_utils.get_filter_fn(min_length, blacklist=blacklist)
    ds = ds.select(filter_fn)
    print("added select filtering...", min_length)

    if shuffle:
        ds = ds.shuffle(100)

    custom_collation_fn = loader_utils.get_custom_collation_fn(
        coordinator_params['k_neighbors'])
    batched_ds = ds.batched(args.batch_size, collation_fn=custom_collation_fn)
    dl = torch.utils.data.DataLoader(batched_ds, num_workers=args.num_workers, batch_size=None)
    count = 0
    if return_count:
        for u in tqdm.tqdm(dl):
            count += 1
    return dl, count

def get_wds_loaders(args, coordinator_params, gpu=None, shuffle_train=True, val_only=False, return_count=True):
    #assert args.distributed == 0
    train_path = os.path.join(args.data_root, args.train_wds_path)
    val_path = os.path.join(args.data_root, args.val_wds_path)
    train_blacklist = get_blacklist(args)
    if args.zip_enabled == 1:
        if not val_only:
            dist_args = None
            if args.distributed == 1:
                dist_args = {'world_size': args.world_size, 'rank': gpu}
            train_dl, train_count = create_ziploader(
                args, coordinator_params, train_path, shuffle=shuffle_train, return_count=return_count,
                dist_args=dist_args)
        else:
            train_dl, train_count = None, 0
        val_dl, val_count = create_ziploader(
                args, coordinator_params, val_path, shuffle=False, return_count=return_count,
                dist_args=dist_args, is_val=True)
    else:
        if args.distributed == 1:
            if not val_only:
                train_dl, train_count = create_distributed_wds_loader(
                    args, coordinator_params, train_path, shuffle=shuffle_train, count=args.dist_train_len, blacklist=train_blacklist)
            else:
                train_dl, train_count = None, 0
            val_dl, val_count = create_distributed_wds_loader(
                args, coordinator_params, val_path, shuffle=False, count=args.dist_val_len)
        else:
            if not val_only:
                train_dl, train_count = create_simple_wds_loader(
                    args, coordinator_params, train_path, shuffle=shuffle_train, return_count=return_count, blacklist=train_blacklist)
            else:
                train_dl, train_count = None, 0
            val_dl, val_count = create_simple_wds_loader(args, coordinator_params, val_path, shuffle=False, return_count=return_count)
    print(train_count, val_count)
    return train_dl, val_dl, train_count, val_count

def main(gpu, config_args, exp_name, logpath):
    args = config_args
    if args.distributed:
        assert gpu is not None
    training_device = gpu if args.distributed else 'cuda'
        
    training_args = {'epochs': args.epochs, 
                     'lr': args.lr,
                     'weight_decay': args.weight_decay, 
                     'momentum': args.momentum,
                     'label_smoothing': args.label_smoothing,
                     'lr_peak_epoch': args.lr_peak_epoch,
                     'eval_epochs': args.eval_epochs,
                     'only_non_bn_weight_decay':args.only_non_bn_weight_decay,
                     'opt': args.opt, 
                     'lr_scheduler': args.lr_scheduler,
                     'burn_in': args.burn_in,
                     'mixed_precision': args.mixed_precision == 1,
                     'max_len': args.max_seq_len,
                     'self_supervised': args.self_supervised == 1,
                     }
    

    coordinator_params = data_utils.get_coordinator_params(args.coordinator_hparams)
    coordinator_params['num_positional_embeddings'] = args.gnn_num_pos_embs
    coordinator_params['zero_out_pos_embs']= args.gnn_zero_out_pos_embs
    coordinator_params['clip_mode'] = True
    gnn_checkpoint = args.gnn_checkpoint
    if gnn_checkpoint == '':
        gnn_checkpoint = None
    model_building_args = {'esm_arch': args.arch, 
                           'terminator_hparams': coordinator_params,
                           'self_supervised': args.self_supervised == 1,
                           'gnn_checkpoint': gnn_checkpoint,
                           'freeze_llm': args.freeze_llm == 1,
                           'freeze_text_proj': args.freeze_text_proj == 1,
                           'use_text_proj': args.use_text_proj == 1,
                           'projection_dim': args.projection_dim,
                           'language_head': args.language_head == 1,
                           'language_head_type': args.language_head_type, 
                          }
    if args.finetune_from == '':
        model = create_clip_model(args.arch, model_building_args, device=training_device)
    else:
        print("finetuning from", args.finetune_from)
        model = load_model(args.finetune_from, device=training_device)

    tokenizer = EsmTokenizer.from_pretrained(args.arch)
    # require grad out unused params ---> this is not enough for distributed training!
    zero_grad_params = [model.text_model.pooler, model.text_model.contact_head]
    if model.text_model.embeddings.position_embedding_type == 'rotary':
        zero_grad_params.append(model.text_model.embeddings.position_embeddings)
    if args.freeze_llm:
        # freeze the entire esm model
        zero_grad_params.append(model.text_model)
    if args.freeze_text_proj:
        zero_grad_params.append(model.text_projection)
    for P in zero_grad_params:
        for name, p in P.named_parameters():
            p.requires_grad = False

    if args.mixed_precision == 0:
        model = model.float()
    if args.distributed:
        model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        
    train_loader, val_loader, train_len, val_len = get_wds_loaders(args, coordinator_params)
    
    trainer = LightWeightTrainer(training_args, exp_name, logpath=logpath,
                                 enable_logging=not args.disable_logging,
                                 model_building_args=model_building_args,
                                 device=training_device,
                                 tokenizer=tokenizer,
                                 train_len=train_len, 
                                 val_len=val_len,
                                 num_mutations=args.num_mutations,
                                 mutation_loss_fwd_method=args.mutation_fwd_method,
                                 distributed=args.distributed == 1,
                                 lm_loss=args.language_head == 1,
                                 lm_weight=args.lm_weight,
                                 resid_weight=args.resid_weight,
                                 min_length=coordinator_params['k_neighbors'],
                                 lm_only_text=args.lm_only_text == 1,
                                )

    trainer.fit(model, train_dataloader=train_loader, val_dataloader=val_loader)

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    args = config_parse_utils.process_args_and_config()
    data_root = args.data_root            
    
    id_str = str(uuid.uuid4())
    EXP_NAME = id_str if not args.exp_name else args.exp_name
    log_path = dist_utils.make_training_dir(args.out_dir, EXP_NAME)
    
    pkl_log_path = log_path
    if pkl_log_path is None:
        pkl_log_path = os.path.join(args.out_dir, EXP_NAME)
        os.makedirs(pkl_log_path, exist_ok=True)
    
    all_out = {'args': vars(args)}
    torch.save(all_out, os.path.join(pkl_log_path, id_str + '.pt'))
    print(args)
    
    distributed = args.distributed
    if distributed:
        dist_manager = dist_utils.DistributedManager(world_size=args.world_size, address=args.address, port=args.port)
        dist_manager.launch_from_args(main, cargs=[args, EXP_NAME, log_path])
    else:
        main('cuda', config_args=args, exp_name=EXP_NAME, logpath=log_path)



    print("==>[Job successfully done.]")
