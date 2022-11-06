import sys
import math

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

from utils.utils import shuffle_batch, shuffle_instance
from architecture.transformer import Transformer, pos_enc_1d

class IPSNet(nn.Module):
    ''' Net that runs IPS, patch encoder, aggregator, class. head '''

    def get_patch_enc(self, enc_type, pretrained, n_chan_in, n_res_blocks):
        # get architecture for patch encoder
        if enc_type == 'resnet18': 
            res_net_fn = resnet18
        elif enc_type == 'resnet50':
            res_net_fn = resnet50
        res_net = res_net_fn(pretrained=pretrained)

        if n_chan_in == 1:
            # standard resnet uses 3 input channels
            res_net.conv1 = nn.Conv2d(n_chan_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # compose patch encoder
        layer_ls = []
        layer_ls.extend([
            res_net.conv1,
            res_net.bn1,
            res_net.relu,
            res_net.maxpool,
            res_net.layer1,
            res_net.layer2
        ])

        if n_res_blocks == 4:
            layer_ls.extend([
                res_net.layer3,
                res_net.layer4
            ])
        
        layer_ls.append(res_net.avgpool)

        return nn.Sequential(*layer_ls)

    def get_output_layers(self, tasks):
        # define output layer for each task

        D = self.D
        n_class = self.n_class

        output_layers = nn.ModuleDict()
        for task in tasks.values():
            if task['act_fn'] == 'softmax':
                torch_act_fn = nn.Softmax(dim=-1)
            elif task['act_fn'] == 'sigmoid':
                torch_act_fn = nn.Sigmoid()
            
            layers = [
                nn.Linear(D, n_class),
                torch_act_fn
            ]
            output_layers[task['name']] = nn.Sequential(*layers)

        return output_layers

    def __init__(self, device, conf):
        super().__init__()

        self.device = device
        self.n_class = conf.n_class
        self.M = conf.M
        self.I = conf.I
        self.D = conf.D 
        self.use_pos = conf.use_pos
        self.tasks = conf.tasks
        self.shuffle = conf.shuffle
        self.shuffle_style = conf.shuffle_style

        if conf.use_patch_enc:
            self.patch_encoder = self.get_patch_enc(conf.enc_type, conf.pretrained,
                conf.n_chan_in, conf.n_res_blocks)

        # define the multi-head cross-attention transformer
        self.transf = Transformer(conf.n_token, conf.H, conf.D, conf.D_k, conf.D_v,
            conf.D_inner, conf.attn_dropout, conf.dropout)

        if conf.use_pos:
            self.pos_enc = pos_enc_1d(conf.D, conf.N).unsqueeze(0).to(device)
        else:
            self.pos_enc = None
        
        # define output layer(s)
        self.output_layers = self.get_output_layers(conf.tasks)

    def do_shuffle(self, patches, pos_enc):
        """ shuffles patches and pos_enc so that patches that have an equivalent score
            are sampled uniformly """

        shuffle_style = self.shuffle_style
        if shuffle_style == 'batch':
            patches, shuffle_idx = shuffle_batch(patches)
            if torch.is_tensor(pos_enc):
                pos_enc, _ = shuffle_batch(pos_enc, shuffle_idx)
        elif shuffle_style == 'instance':
            patches, shuffle_idx = shuffle_instance(patches, 1)
            if torch.is_tensor(pos_enc):
                pos_enc, _ = shuffle_instance(pos_enc, 1, shuffle_idx)
        
        return patches, pos_enc

    def score_and_select(self, emb, emb_pos, M, idx):
        # scores embeddings and selects the top-M embeddings
        B, D = emb.shape[0], emb.shape[2]

        emb_to_score = emb_pos if torch.is_tensor(emb_pos) else emb

        attn = self.transf.get_scores(emb_to_score)

        top_idx = torch.topk(attn, M, dim = -1)[1]
        
        mem_emb = torch.gather(emb, 1, top_idx.unsqueeze(-1).expand(-1,-1,D))
        mem_idx = torch.gather(idx, 1, top_idx)

        return mem_emb, mem_idx

    def get_preds(self, embeddings):
            preds = {}
            for task in self.tasks.values():
                t_name, t_id = task['name'], task['id']
                layer = self.output_layers[t_name]

                emb = embeddings[:,t_id]
                preds[t_name] = layer(emb)            

            return preds

    @torch.no_grad()
    def ips(self, patches):
        # get useful info
        M = self.M
        I = self.I
        D = self.D  
        device = self.device
        shuffle = self.shuffle
        use_pos = self.use_pos
        pos_enc = self.pos_enc
        patch_shape = patches.shape
        B, N = patch_shape[:2]

        # check if IPS required
        if M >= N:
            pos_enc = pos_enc.expand(B, -1, -1) if use_pos else None
            return patches.to(device), pos_enc 

        # should patch encoder be used?
        if len(patch_shape) == 3: # B, N, D
            is_image = False
        elif len(patch_shape) == 5: # B, N, n_chan_in, height, width
            is_image = True
        else:
            raise ValueError('The input is neither an image (5 dim) nor a feature vector (3 dim).')
        
        # IPS runs in evaluation mode
        if self.training:
            self.patch_encoder.eval()
            self.transf.eval()

        # adjust positional encoding to batch
        if use_pos:
            pos_enc = pos_enc.expand(B, -1, -1)

        # shuffle patches
        if shuffle:
            patches, pos_enc = self.do_shuffle(patches, pos_enc)

        # init memory buffer
        init_patch = patches[:,:M].to(device)
        if is_image:
            mem_emb = self.patch_encoder(init_patch.reshape(-1, *patch_shape[2:]))
            mem_emb = mem_emb.view(B, M, -1)
        else:
            mem_emb = init_patch
        
        # init mem idx
        idx = torch.arange(N, dtype=torch.int64, device=device).unsqueeze(0).expand(B, -1)
        mem_idx = idx[:,:M]

        # get num iterations
        n_iter = math.ceil((N - M) / I)
        for i in range(n_iter):
            # get next patches
            start_idx = i * I + M
            end_idx = min(start_idx + I, N)

            iter_patch = patches[:, start_idx:end_idx].to(device)
            iter_idx = idx[:, start_idx:end_idx]

            # embed patches
            if is_image:
                iter_emb = self.patch_encoder(iter_patch.reshape(-1, *patch_shape[2:]))
                iter_emb = iter_emb.view(B, -1, D)
            else:
                iter_emb = iter_patch
            
            # concatenate with memory buffer
            all_emb = torch.cat((mem_emb, iter_emb), dim=1)
            all_idx = torch.cat((mem_idx, iter_idx), dim=1)
            if use_pos:
                all_pos_enc = torch.gather(pos_enc, 1, all_idx.view(B, -1, 1).expand(-1, -1, D))
                all_emb_pos = all_emb + all_pos_enc
            else:
                all_emb_pos = None

            mem_emb, mem_idx = self.score_and_select(all_emb, all_emb_pos, M, all_idx)

        # select patches
        n_dim_expand = len(patch_shape) - 2
        mem_patch = torch.gather(patches, 1, 
            mem_idx.view(B, -1, *(1,)*n_dim_expand).expand(-1, -1, *patch_shape[2:]).to(patches.device)
        ).to(device)

        if use_pos:
            mem_pos = torch.gather(pos_enc, 1, mem_idx.unsqueeze(-1).expand(-1, -1, D))
        else:
            mem_pos = None

        # set network back to training mode            
        if self.training:
            self.patch_encoder.train()
            self.transf.train()
    
        return mem_patch, mem_pos

    def forward(self, mem_patch, mem_pos=None):
        patch_shape = mem_patch.shape
        B, M = patch_shape[:2]

        if len(patch_shape) == 3: # B, N, D
            is_image = False
        elif len(patch_shape) == 5: # B, N, n_chan_in, height, width
            is_image = True
        else:
            raise ValueError('The input is neither an image (5 dim) nor a feature vector (3 dim).')
        
        if is_image:
            mem_emb = self.patch_encoder(mem_patch.reshape(-1, *patch_shape[2:]))
            mem_emb = mem_emb.view(B, M, -1)        

        if torch.is_tensor(mem_pos):
            mem_emb = mem_emb + mem_pos

        image_emb = self.transf(mem_emb)

        preds = self.get_preds(image_emb)
        
        return preds