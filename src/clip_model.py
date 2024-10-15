import torch
import torch.nn as nn
from terminator.models.TERMinator import TERMinator
from transformers import EsmModel
import math

def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        
class LMHead(nn.Module):
    """ESM Head for masked language modeling."""

    def __init__(self, hidden_size, vocab_size=33):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-05)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x
    
def transform_pretrained_gnn(state_dict):
    replace_list = [
        ['top.features', 'top.featurizer.features'],
        ['top.W_v', 'top.featurizer.W_v'],
        ['top.W_e', 'top.featurizer.W_e'],
    ]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix, new_prefix in replace_list:
            if new_k.startswith(prefix):
                new_k = new_prefix + new_k[len(prefix):]
        new_state_dict[new_k] = v
    return new_state_dict

class SmallLMHead(nn.Module):
    """ESM Head for masked language modeling."""

    def __init__(self, hidden_size, vocab_size=33):
        super().__init__()
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, features):
        # project back to size of vocabulary with bias
        x = self.decoder(features) + self.bias
        return x

class ProteinCLIP(nn.Module):
    def __init__(self, esm_arch, terminator_hparams, gnn_checkpoint=None, 
                 projection_dim=320,
                 self_supervised=True, freeze_llm=False, 
                 freeze_text_proj=False,
                 lm_head_text=False,
                 lm_head_type='MLP',
                 device='cuda:0',
                 use_text_proj=True,
                 ):
        super().__init__()
        self.esm_arch = esm_arch
        self.terminator_hparams = terminator_hparams
        gnn_emb_dim = terminator_hparams['energies_hidden_dim']
        text_emb_dim = 640  
        self.text_model = EsmModel.from_pretrained(esm_arch) 

        self.gnn_model = TERMinator(hparams=terminator_hparams, device=device)
        if gnn_checkpoint is not None:
            print("loading pretrained gnn", gnn_checkpoint)
            gnn_state = transform_pretrained_gnn(torch.load(gnn_checkpoint)['state_dict'])
            self.gnn_model.load_state_dict(gnn_state, strict=False)
        
        self.gnn_postprocess = nn.Sequential(
            nn.Linear(gnn_emb_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.gnn_projection = nn.Linear(projection_dim, projection_dim, bias=False)

        self.text_projection = nn.Linear(text_emb_dim, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        self.self_supervised = self_supervised
        print("freeze_llm", freeze_llm)
        print("use text proj: ", use_text_proj)
        self.use_text_proj = use_text_proj
        self.freeze_llm = freeze_llm
        self.freeze_text_proj = freeze_text_proj
        if lm_head_text:
            if lm_head_type == 'MLP':
                self.lm_head = LMHead(projection_dim)
            elif lm_head_type == 'LINEAR':
                self.lm_head = SmallLMHead(projection_dim)
            else:
                assert False
        else:
            self.lm_head = None

    def loadGNNModel(self, gnn_checkpoint):
        print("loading pretrained gnn", gnn_checkpoint)
        gnn_state = torch.load(gnn_checkpoint)['state_dict']
        self.gnn_model.load_state_dict(gnn_state)
    
    def get_GNN(self):
        return self.gnn_model

    def get_avg(self, enc, mask):
        enc[mask == 0] = 0
        if torch.isnan(enc).any().item():
            import ipdb
            ipdb.set_trace()
        return enc.sum(1)/mask.sum(1).unsqueeze(1)
        #return (mask.unsqueeze(1).float() @ enc).squeeze(1) / mask.sum(1, keepdims=True)

    def get_text_features(self, text_inp):
        if self.freeze_llm:
            with torch.no_grad():
                output = self.text_model(**text_inp)
        else:
            output = self.text_model(**text_inp)
        hidden_states = output.last_hidden_state
        if self.self_supervised:
            enc = hidden_states
        else:
            enc = self.get_avg(hidden_states, text_inp['attention_mask'])
            #enc = hidden_states[:, 0]
            
        if self.freeze_text_proj:
            with torch.no_grad():
                text_feats = self.text_projection(enc)
        else:
            if self.use_text_proj:
                text_feats = self.text_projection(enc)
            else:
                text_feats = enc
        return text_feats
    
    def get_text_features_no_proj(self, text_inp):
        if self.freeze_llm:
            with torch.no_grad():
                output = self.text_model(**text_inp)
        else:
            output = self.text_model(**text_inp)
        hidden_states = output.last_hidden_state
        if self.self_supervised:
            enc = hidden_states
        else:
            enc = self.get_avg(hidden_states, text_inp['attention_mask'])
            #enc = hidden_states[:, 0]
        return enc
    
    def get_graph_features(self, coord_data):
        _, _, embs = self.gnn_model(coord_data)
        output = self.gnn_postprocess(embs)
        if not self.self_supervised:
            output = self.get_avg(output, coord_data['x_mask'])
            #output = (coord_data['x_mask'].unsqueeze(1) @ output).squeeze(1)
        gnn_feats = self.gnn_projection(output)
        return gnn_feats

    def get_lm_output(self, embeds):
        return self.lm_head(embeds)
        
    def forward(self, text_inp, coord_data):
        text_embeds, gnn_embeds = None, None
        if text_inp is not None:
            text_embeds = self.get_text_features(text_inp)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        if coord_data is not None:
            gnn_embeds = self.get_graph_features(coord_data)
            gnn_embeds = gnn_embeds / gnn_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        return gnn_embeds, text_embeds, logit_scale