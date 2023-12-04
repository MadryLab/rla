import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_weighted_metric(mask, loss):
    b = mask.shape[0]
    loss_weight = mask.sum(dim=1, keepdims=True).expand(mask.shape)[mask==1]
    loss = (loss[mask==1]/loss_weight).sum()/b
    return loss     

class CLIPLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
    def forward(self, gnn_features, text_features, logit_scale):
        loss_one = logit_scale * gnn_features @ text_features.T # N x N
        loss_two = logit_scale * text_features @ gnn_features.T # N x N
        labels = torch.arange(loss_one.shape[-1]).long().to(self.device)
        loss = (F.cross_entropy(loss_one, labels) + F.cross_entropy(loss_two, labels)) / 2
        acc = ((loss_one.argmax(-1) == labels).float().mean() + (loss_two.argmax(-1) == labels).float().mean())/2
        return loss, acc
       
class LMLoss(nn.Module):
    def __init__(self, only_text=True, device='cuda'):
        super().__init__()
        self.only_text = only_text
        self.device = device

    def sub_loss(self, lm_mask, lm_out, wt_input_ids):
        loss =  F.cross_entropy(lm_out.permute(0, 2, 1), wt_input_ids, reduction='none')
        loss = compute_weighted_metric(lm_mask, loss)
        acc = (lm_out.argmax(-1) == wt_input_ids).float()
        acc = compute_weighted_metric(lm_mask, acc).item()
        return loss, acc

    def forward(self, lm_mask, text_out, wt_input_ids, 
                gnn_out, coords_to_seq, coords_loss_mask):
        text_loss, text_acc = self.sub_loss(lm_mask=lm_mask, lm_out=text_out, wt_input_ids=wt_input_ids)
        if self.only_text:
            gnn_loss, gnn_acc = 0, 0
        else:
        # gnn
            coord_indexing = coords_to_seq.to(self.device).clone()
            coord_indexing[coords_loss_mask == False] = 0
            coords_loss_mask = coords_loss_mask.to(self.device)
            gnn_input_ids = torch.gather(wt_input_ids, 1, coord_indexing)
            gnn_lm_mask = torch.gather(lm_mask, 1, coord_indexing) & coords_loss_mask
            gnn_loss, gnn_acc = self.sub_loss(lm_mask=gnn_lm_mask, lm_out=gnn_out, wt_input_ids=gnn_input_ids)
        return text_loss, text_acc, gnn_loss, gnn_acc


class CLIPSelfSupLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

    def sub_loss(self, out, index, loss_mask):
        B = loss_mask.shape[0]
        out, index = out[loss_mask], index[loss_mask]
        loss_weights = loss_mask.sum(dim=1, keepdims=True).expand(loss_mask.shape)[loss_mask]
        loss = F.cross_entropy(out, index, reduction='none')
        loss = loss/loss_weights
        loss = loss.sum()/B
        acc = (out.argmax(1) == index).float().mean()
        return loss, acc

        
    def forward(
        self, gnn_features, text_features, logit_scale, seq_to_coords, 
        seq_loss_mask, coords_to_seq, coords_loss_mask
        ):
        out = logit_scale*(gnn_features @ text_features.permute(0, 2, 1))
        out[coords_loss_mask == False] = -torch.inf
        out.transpose(1, 2)[seq_loss_mask == False] = -torch.inf
        
        seq_to_coords = seq_to_coords.to(self.device)
        coords_to_seq = coords_to_seq.to(self.device)
        seq_loss_mask = seq_loss_mask.to(self.device)
        coords_loss_mask = coords_loss_mask.to(self.device)
        loss_one, acc_one = self.sub_loss(
            out=out.transpose(1,2), index=seq_to_coords, loss_mask=seq_loss_mask)
        loss_two, acc_two = self.sub_loss(
            out=out, index=coords_to_seq, loss_mask=coords_loss_mask)
        loss = (loss_one + loss_two)/2
        acc = (acc_one + acc_two)/2
        return loss, acc  

class CLIPMutationLoss(nn.Module):
    def __init__(self, num_mutations, device='cuda', forward_method='all'):
        super().__init__()
        self.device = device
        self.num_mutations = num_mutations
        self.forward_method = forward_method
        
    def forward(
        self, gnn_features, text_features, logit_scale, seq_to_coords, 
        coord_to_change, seq_loss_mask
        ):
        if self.forward_method == 'single':
            # only supervise the residue which changed
            return self.forward_single(
                gnn_features=gnn_features,
                text_features=text_features,
                logit_scale=logit_scale,
                seq_to_coords=seq_to_coords,
                coord_to_change=coord_to_change,
            )
        else:
            # supervise all the residues
            return self.forward_all(
                gnn_features=gnn_features,
                text_features=text_features,
                logit_scale=logit_scale,
                seq_to_coords=seq_to_coords,
                seq_loss_mask=seq_loss_mask,
            )
        return loss, acc

    def forward_single(
        self, gnn_features, text_features, logit_scale, seq_to_coords, 
        coord_to_change
        ):
        B = len(gnn_features)
        Brange = torch.arange(B)
        unflattened = text_features.unflatten(0, (B, self.num_mutations+1))
        selected = unflattened[Brange, :, coord_to_change]
        gnn_inds = seq_to_coords[Brange, coord_to_change]
        selected_gnn = gnn_features[Brange, gnn_inds].unsqueeze(1)
        
        scores = (selected.unsqueeze(2)  @ selected_gnn.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        scores = scores * logit_scale
        correct = torch.zeros(B).to(self.device).long()
        loss = F.cross_entropy(scores, correct)
        acc = (scores.argmax(1) == 0).float().mean()
        return loss, acc

    def forward_all(
        self, gnn_features, text_features, logit_scale, seq_to_coords, 
        seq_loss_mask
        ):
        seq_loss_mask = seq_loss_mask.to(self.device)
        B = len(gnn_features)
        Brange = torch.arange(B)
        unflattened = text_features.unflatten(0, (B, self.num_mutations+1))
        selected_gnn = torch.stack([gnn_features[b][seq_to_coords[b]] for b in range(B)])
        scores =  (unflattened.unsqueeze(3) @  selected_gnn.unsqueeze(-1).unsqueeze(1)).squeeze(-1).squeeze(-1)
        scores = scores * logit_scale  # N x M x T
        T = scores.shape[-1]
        correct = torch.zeros((B, T)).to(self.device).long()
        loss = F.cross_entropy(scores, correct, reduction='none')
        loss_weights = seq_loss_mask.sum(dim=1, keepdims=True).expand(seq_loss_mask.shape)[seq_loss_mask]
        loss = loss[seq_loss_mask]/loss_weights
        loss = loss.sum()/B
        acc = (scores.transpose(1,2).argmax(2)[seq_loss_mask] == 0).float().mean()
        return loss, acc