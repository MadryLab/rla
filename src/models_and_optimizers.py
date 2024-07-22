from torch.cuda.amp import GradScaler
import torch as ch
import numpy as np
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR
import torchvision.models as torch_models
import open_clip
import torch
from src.clip_model import ProteinCLIP

def save_model(model, path, run_metadata):
    torch.save({
        'state_dict': model.state_dict(),
        'run_metadata': run_metadata
    }, path)
    
def _unwrap_ddp_model(state_dict):

    sub_dict = {
        'module.':'',
        'gnn_model.top.features': 'gnn_model.top.featurizer.features',
        'gnn_model.top.W_v': 'gnn_model.top.featurizer.W_v',
        'gnn_model.top.W_e': 'gnn_model.top.featurizer.W_e',
    }
    new_state_dict = {}
    for k, v in state_dict.items():
        for old, new in sub_dict.items():
            if k.startswith(old):
                k = new + k[len(old):]
        new_state_dict[k] = v
    return new_state_dict

def load_model(path, esm_arch, device, coordinator_checkpoint=None, model=None, load_state_dict=True):
    print("loading state dict from", path)
    ckpt = torch.load(path, map_location=torch.device(device))
    if model is None: # build based on path
        print("building model based on path")
        model_building_args = ckpt['run_metadata']['model_building_args']
        model = create_clip_model(esm_arch, model_building_args, device=device, coordinator_checkpoint=coordinator_checkpoint)
    if load_state_dict:
        state_dict = _unwrap_ddp_model(ckpt['state_dict'])
        if 'text_model.embeddings.position_ids' not in state_dict.keys():
            state_dict['text_model.embeddings.position_ids'] = torch.arange(1026).unsqueeze(0).to(device)
        model.load_state_dict(state_dict)
    return model

def create_clip_model(esm_arch, model_building_args, device, coordinator_checkpoint=None):
    if coordinator_checkpoint is not None:
        model_building_args['gnn_checkpoint'] = coordinator_checkpoint
    model = ProteinCLIP(esm_arch=esm_arch, 
                        gnn_checkpoint=model_building_args['gnn_checkpoint'],
                        terminator_hparams=model_building_args['terminator_hparams'],
                        projection_dim=model_building_args.get('projection_dim', 640),
                        self_supervised=model_building_args['self_supervised'],
                        freeze_llm=model_building_args.get('freeze_llm', False),
                        use_text_proj=model_building_args.get('use_text_proj', False),
                        lm_head_text=model_building_args.get('language_head', False),
                        lm_head_type=model_building_args.get('language_head_type', 'MLP'),
                        device=device
                       )
    print('args: ')
    print(model_building_args.get('projection_dim', 640), model_building_args.get('use_text_proj', False))
    model = model.to(memory_format=ch.channels_last)
    model = model.to(device)
    
    return model


class LRPolicy():
    def __init__(self, lr_schedule):
        self.lr_schedule = lr_schedule
    def __call__(self, epoch):
        return self.lr_schedule[epoch]
    

# https://github.com/mlfoundations/open_clip/blob/6e6c2473c01aa400da1d972fa03840abc7fd3fbb/src/training/scheduler.py

class CosineLR():
    def __init__(self, optimizer, base_lr, warmup_length, steps):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_length = warmup_length
        self.steps = steps
        self._step = 0
        self.last_lr = base_lr
        
    def _warmup_lr(self):
        return self.base_lr * (self._step + 1) / self.warmup_length

    def _assign_learning_rate(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def step(self):
        if self._step < self.warmup_length:
            lr = self._warmup_lr()
        else:
            e = self._step - self.warmup_length
            es = self.steps - self.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * self.base_lr
        self._assign_learning_rate(lr)
        self.last_lr = lr
        self._step += 1
        return lr
    
    def get_last_lr(self):
        return [self.last_lr]
    

def get_optimizer_and_lr_scheduler(training_args, model, iters_per_epoch=1):
    only_non_bn_weight_decay = training_args['only_non_bn_weight_decay']
    weight_decay = training_args['weight_decay']
    
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    
    param_groups = [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": weight_decay},
    ]
    
    epochs = training_args['epochs']
    
    if training_args['opt'] == 'SGD':
        optimizer = SGD(param_groups, lr=training_args['lr'], 
                        momentum=training_args['momentum'], weight_decay=weight_decay)
    elif training_args['opt'] == 'ADAM':
        optimizer = AdamW(param_groups,  
                          lr=training_args['lr'], 
                          weight_decay=weight_decay,
                          betas=(0.9, 0.999), eps=1.0e-8,
                         )
    else:
        assert False
    
    if training_args['lr_scheduler'] == 'cyclic':
        lr_peak_epoch = training_args['lr_peak_epoch']        
        lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                        [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                        [0, 1, 0])
        scheduler = LambdaLR(optimizer, LRPolicy(lr_schedule)) 
    elif training_args['lr_scheduler'] == 'cosine':
        scheduler = CosineLR(optimizer, training_args['lr'], 
                              iters_per_epoch*training_args['lr_peak_epoch'], steps=epochs*iters_per_epoch)
    else:
        assert False
    return optimizer, scheduler
