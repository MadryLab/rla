import os

from threading import Lock
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from src.models_and_optimizers import get_optimizer_and_lr_scheduler
from src.models_and_optimizers import save_model, load_model
from src.eval_utils import AverageMeter
import src.data_utils as data_utils
import itertools
from src.losses import compute_weighted_metric, CLIPLoss, LMLoss, CLIPSelfSupLoss, CLIPMutationLoss
lock = Lock()

class LightWeightTrainer():
    def __init__(self, training_args, 
                 exp_name, 
                 logpath,
                 tokenizer,
                 enable_logging=True, 
                 model_building_args=None,
                 device='cuda',
                 train_len=None,
                 val_len=None,
                 max_len=1024,
                 num_mutations=6,
                 distributed=False,
                 mutation_loss_fwd_method='all',
                 lm_loss=False,
                 lm_weight=1,
                 resid_weight=1,
                 min_length=None,
                 lm_only_text=True,
                ):
        # training args
        self.device = device
        self.is_logging_device = (device == 0 or device == 'cuda')
        self.training_args = training_args
        self.check_training_args_()
        self.tokenizer = tokenizer
        
        self.self_supervised = training_args['self_supervised']
        self.max_len = training_args['max_len']
        self.mixed_precision = training_args['mixed_precision']
        self.burn_in = training_args['burn_in']
        self.distributed = distributed
        # loss
        if self.self_supervised:
            self.clip_loss = CLIPSelfSupLoss(device=device)
        else:
            self.clip_loss = CLIPLoss(device=device)
       

        # logging
        self.enable_logging = enable_logging

        if self.enable_logging:    
            self.writer = SummaryWriter(logpath)
            self.training_dir = logpath
            self.model_building_args = model_building_args
        else:
            self.training_dir = None
        self.train_len = train_len
        self.val_len = val_len
        self.num_mutations = num_mutations
        if self.num_mutations > 0:
            self.mutation_loss = CLIPMutationLoss(num_mutations=self.num_mutations, device=device, 
            forward_method=mutation_loss_fwd_method)

        if lm_loss:
            self.lm_loss = LMLoss(device=device, only_text=lm_only_text)
        else:
            self.lm_loss = None
            
        self.lm_weight = lm_weight
        self.resid_weight = resid_weight
        self.min_length=min_length
        
    def check_training_args_(self):
        for z in ['epochs', 'lr', 'weight_decay', 'momentum', 'only_non_bn_weight_decay',
                  'lr_peak_epoch', 'eval_epochs', 'label_smoothing', 'opt', 'lr_scheduler',
                  'max_len', 'self_supervised', 'burn_in'
                 ]:
            assert z in self.training_args, f'{z} not in training_args'



    def get_accuracy(self, logits, target):
        correct = logits.argmax(-1) == target
        return (correct.float().mean()) * 100

    def get_opt_scaler_scheduler(self, model, iters_per_epoch=1):
        opt, scheduler = get_optimizer_and_lr_scheduler(self.training_args, model, iters_per_epoch)
        scaler = GradScaler(enabled=self.mixed_precision)
        return opt, scaler, scheduler

    def _tokenize(self, seqs, pos_embs):
        text_inp = self.tokenizer(seqs, return_tensors='pt', padding=True, 
                                  truncation=True, max_length=self.max_len+2)
        text_inp['position_ids'] = pos_embs
        text_inp = {k: v.to(self.device) for k, v in text_inp.items()}
        return text_inp

    def step(self, model, batch):
        seq_batch, coords_batch = batch
        do_indiv_mutations = 'mutation_seqs' in seq_batch
        seqs = seq_batch['string_sequence']
        pos_embs = seq_batch['pos_embs'][0]
        pl_mask = seq_batch['placeholder_mask'][0]
        llm_masking = 'llm_masked_sequence' in seq_batch

        # set up individual mutation loss
        if do_indiv_mutations:
            seqs = list(itertools.chain.from_iterable(seq_batch['mutation_seqs']))
            coord_to_change = torch.stack(seq_batch['coord_to_change'])
            pos_embs = pos_embs.unsqueeze(1).expand(-1, self.num_mutations+1, -1).flatten(0,1)
            pl_mask = pl_mask.unsqueeze(1).expand(-1, self.num_mutations+1, -1).flatten(0,1)

        # llm masked elements
        if llm_masking:
            assert not do_indiv_mutations, "can't do both llm masking and mutations"
            seqs = seq_batch['llm_masked_sequence']

        # set up inputs
        text_inp = self._tokenize(seqs, pos_embs)

        if llm_masking:
            gt_text_inp = self._tokenize(seq_batch['string_sequence'], pos_embs)
        else:
            gt_text_inp = text_inp

        coord_data  = data_utils.construct_gnn_inp(coords_batch, device=self.device,
                                                   half_precision=self.mixed_precision)
        # get features
        with lock:
            gnn_features, text_features, logit_scale = model(text_inp, coord_data)
            
        generic_metrics = {}
        
        ##==============
        ## Residue level loss
        ##==============
        if self.self_supervised:
            # self supervised loss
            new_text_features, new_input_ids, new_text_mask = data_utils.postprocess_text_features(
                text_features=text_features, inp_dict=gt_text_inp, 
                tokenizer=self.tokenizer,  placeholder_mask=pl_mask,
                min_length=self.min_length,
            )

            # set defaults
            wt_text_features = new_text_features # wild type text features
            wt_input_ids = new_input_ids # wild type input ids
            indiv_mut_loss, text_loss = 0, 0

            # ==============
            # invidual mutations loss
            # ==============
            if do_indiv_mutations:
                batch_len = len(seq_batch['string_sequence'])
                wt_indices = torch.arange(batch_len)*(self.num_mutations+1)
                wt_text_features = new_text_features[wt_indices]
                wt_input_ids = new_input_ids[wt_indices]
                indiv_mut_loss, indiv_mut_acc = self.mutation_loss(
                    gnn_features=gnn_features,
                    text_features=new_text_features,
                    logit_scale=logit_scale, 
                    seq_to_coords=seq_batch['seq_to_coords'][0],
                    coord_to_change=coord_to_change,
                    seq_loss_mask=seq_batch['seq_loss_mask'][0],
                )
                generic_metrics['mut_loss']  = indiv_mut_loss.item()
                generic_metrics['mut_acc'] = indiv_mut_acc.item()
            
            # ==============
            # language model loss
            # ==============
            if self.lm_loss is not None:
                with lock:
                    lm_mask = seq_batch['llm_mask'][0].to(self.device) if 'llm_mask' in seq_batch else new_text_mask
                    lm_text_out = model.get_lm_output(wt_text_features)
                    lm_gnn_out = model.get_lm_output(gnn_features)
                    lm_text_loss, lm_text_acc, lm_gnn_loss, lm_gnn_acc = self.lm_loss(
                        lm_mask=lm_mask, text_out=lm_text_out, 
                        wt_input_ids=wt_input_ids, gnn_out=lm_gnn_out,
                        coords_to_seq=coords_batch['coords_to_seq'][0],
                        coords_loss_mask=coords_batch['coords_loss_mask'][0])
                    text_loss = (lm_text_loss + lm_gnn_loss) * self.lm_weight
                    generic_metrics['lm_text_loss'] = lm_text_loss.item() * self.lm_weight
                    generic_metrics['lm_text_acc'] = lm_text_acc
                    if lm_gnn_loss != 0:
                        generic_metrics['lm_gnn_loss'] = lm_gnn_loss.item() * self.lm_weight
                        generic_metrics['lm_gnn_acc'] = lm_gnn_acc

            # ==============
            # regular residue loss
            # ==============
            residue_loss, residue_acc = self.clip_loss(
                gnn_features=gnn_features, 
                text_features=wt_text_features, 
                logit_scale=logit_scale, 
                seq_to_coords=seq_batch['seq_to_coords'][0],
                seq_loss_mask=seq_batch['seq_loss_mask'][0],
                coords_to_seq=coords_batch['coords_to_seq'][0],
                coords_loss_mask=coords_batch['coords_loss_mask'][0],
            )
            residue_loss = residue_loss * self.resid_weight
            generic_metrics['res_loss'] = residue_loss.item()
            generic_metrics['res_acc'] = residue_acc.item()

            # combine
            loss = residue_loss + indiv_mut_loss + text_loss
            acc = residue_acc
        else:
            # clip level loss
            if self.distributed:
                all_gnn_features = torch.distributed.nn.all_gather(gnn_features)
                all_text_features = torch.distributed.nn.all_gather(text_features)
                all_gnn_features = torch.cat(all_gnn_features, dim=0)
                all_text_features = torch.cat(all_text_features, dim=0)
            else:
                all_gnn_features, all_text_features = gnn_features, text_features
            loss, acc = self.clip_loss(all_gnn_features, all_text_features, logit_scale)
            generic_metrics['clip_loss'] = loss.item()
            generic_metrics['clip_acc'] = acc.item()

        return loss, acc, generic_metrics, len(text_inp['input_ids'])
        
    def _initialize_meters(self):
        return {
            'loss': AverageMeter(),
            'acc': AverageMeter(),
            'generic': {}
        }
    
    def _update_meters(self, meters, sz, loss, acc, generic_metrics):
        meters['loss'].update(loss.item(), sz)
        meters['acc'].update(acc.item(), sz)
        for k, v in generic_metrics.items():
            if k not in meters['generic']:
                meters['generic'][k] = AverageMeter()
            meters['generic'][k].update(v, sz)
    
    def _calculate_meters(self, meters):
        avg_loss, avg_acc = meters['loss'].calculate(), meters['acc'].calculate()
        avg_generics = {k: v.calculate() for k, v in meters['generic'].items()}
        return avg_loss, avg_acc, avg_generics

    def train_epoch(self, epoch_num, model, train_dataloader, opt, scaler, scheduler):
        model.train()
        meters = self._initialize_meters()
        total = self.train_len if self.train_len is not None else len(train_dataloader)
        with tqdm(train_dataloader, disable=(not self.is_logging_device), total=total) as t:
            t.set_description(f"Train Epoch: {epoch_num}")
            for index, batch in enumerate(t):
                if index == total:
                    break # necessary for distributed training
                opt.zero_grad(set_to_none=True)
                with autocast(enabled=self.mixed_precision, dtype=torch.float16):
                    loss, acc, generic_metrics, sz = self.step(model, batch)
                if self.is_logging_device:
                    postfix_lr = scheduler.get_last_lr()[0]
                    t.set_postfix({'loss': loss.item(), 'acc': acc.item(), 'lr': postfix_lr})
                self._update_meters(
                    meters=meters, sz=sz, loss=loss, acc=acc, generic_metrics=generic_metrics)
                scaler.scale(loss).backward()
                # for name, param in model.named_parameters():
                #     if param.grad is None:
                #         print(name, param.requires_grad)
                scaler.step(opt)
                scaler.update()
                scheduler.step()
        
        avg_loss, avg_acc, avg_generics = self._calculate_meters(meters)
        return avg_loss, avg_acc, avg_generics

    def val_epoch(self, epoch_num, model, val_dataloader):
        model.eval()
        meters = self._initialize_meters()
        with torch.no_grad():
            total = self.val_len if self.val_len is not None else len(val_dataloader)
            with tqdm(val_dataloader, disable=(not self.is_logging_device), total=total) as t:
                t.set_description(f"Val Epoch: {epoch_num}")
                for index, batch in enumerate(t):
                    if index == total:
                        break # necessary for distributed training
                    with autocast(enabled=self.mixed_precision, dtype=torch.float16):
                        loss, acc, generic_metrics, sz = self.step(model, batch)
                    if self.is_logging_device:
                        t.set_postfix({'loss': loss.item(), 'acc': acc.item()})
                    self._update_meters(
                        meters=meters, sz=sz, loss=loss, acc=acc, generic_metrics=generic_metrics)
        avg_loss, avg_acc, avg_generics = self._calculate_meters(meters)
        return avg_loss, avg_acc, avg_generics


    def fit(self, model, train_dataloader, val_dataloader):
        iters_per_epoch = self.train_len if self.train_len is not None else len(train_dataloader) # for wds
        opt, scaler, scheduler = self.get_opt_scaler_scheduler(model, iters_per_epoch=iters_per_epoch)
        best_val_loss = np.inf
        num_epochs = self.training_args['epochs']
        eval_epochs = self.training_args['eval_epochs']
        for epoch in range(num_epochs):
            # should we do val this epoch
            is_val_epoch = (epoch % eval_epochs == 0 and epoch != 0) or (epoch == num_epochs - 1)
            
                            
            # perform epoch
            train_loss, train_acc, train_generics = self.train_epoch(epoch, model, train_dataloader, opt, scaler, scheduler)
            curr_lr = scheduler.get_last_lr()[0]
            
            # logging stuff
            print_str = f"Device: {self.device} LR: {curr_lr}, Train Loss: {train_loss:0.4f}, Train Acc: {train_acc:0.4f}"
            print(f"Device: {self.device}, Train",  train_generics)
            if is_val_epoch:
                val_loss, val_acc, val_generics = self.val_epoch(epoch, model, val_dataloader)
                print_str += f" Val Loss: {val_loss:0.4f}, Val Acc: {val_acc:0.4f}"
                print(f"Device: {self.device}, Val",  val_generics)
            print(print_str)

            # Save Checkpoints
            if self.enable_logging and self.is_logging_device:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Acc/train", train_acc, epoch)
                self.writer.add_scalar("lr", curr_lr, epoch)

                if not is_val_epoch:
                    continue
                
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Acc/val", val_acc, epoch)

                run_metadata = {
                    'training_args': self.training_args,
                    'epoch': epoch,
                    'training_metrics': {'loss': train_loss, 'acc': train_acc},
                    'val_metrics': {'loss': val_loss, 'acc': val_acc},
                    'model_building_args': self.model_building_args,
                }


                checkpoint_folder = os.path.join(self.training_dir, 'checkpoints')
                checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint_latest.pt')
                save_model(model, checkpoint_path, run_metadata)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint_best.pt')
                    save_model(model, checkpoint_path, run_metadata)
                if epoch % 5 == 0: # flush every 5 steps
                    self.writer.flush()
