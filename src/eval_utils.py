# from wandb import Config
import copy
import os
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from threading import Lock
lock = Lock()


class AverageMeter():
    def __init__(self):
        self.num = 0
        self.tot = 0

    def update(self, val, sz):
        self.num += val*sz
        self.tot += sz

    def calculate(self):
        return self.num/self.tot

class ClassAverageMeter(AverageMeter):
    def __init__(self):
        super().__init__()

    def calculate(self):
        return torch.nan_to_num(self.num/self.tot, nan=0.0)

def evaluate_model(model, loader, target=None):
    is_train = model.training
    model.eval().cuda()

    with torch.no_grad():
        softmax = nn.Softmax(dim=-1)
        gts, predictions, raw_logits = [], [], [] # maybe don't want to save all raw logits...
        for x, y in tqdm(loader):
            x, y = x.cuda(), y.cuda()
            with autocast():
                with lock:
                    raw_out = model(x)
                softmax_out = softmax(raw_out)
                max_class = softmax_out.argmax(-1)
                
                predictions.append(max_class.cpu())
                gts.append(y.cpu())
                raw_logits.append(raw_out.cpu())

        result = {
            'gts': torch.cat(gts).half(),
            'preds': torch.cat(predictions).half(),
            'raw_logits': torch.cat(raw_logits).half(),
        }

        acc = (result['gts'] == result['preds']).float().half().mean() * 100
        print("Accuracy: ", acc.item())
        if target is not None: # subselect by target
            result = {k: v[target] for k, v in result.items()}
        result['acc'] = acc

    model.train(is_train)
    return result