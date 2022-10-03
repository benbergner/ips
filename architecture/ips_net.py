import math

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

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
                res_net.layer4,
                res_net.avgpool
            ])

        return nn.Sequential(layer_ls)

    def get_output_layers(task_dict):
        # define output layer for each task
        output_layers = nn.ModuleDict()
        for task in task_dict.values:
            if task['act_fn'] == 'softmax':
                torch_act_fn = nn.Softmax(dim=-1)
            elif task['act_fn'] == 'sigmoid':
                torch_act_fn = nn.Sigmoid()
            
            output_layers[task['name']] = nn.Sequential(
                nn.Linear(D, C),
                torch_act_fn
            )
        return output_layers

    def __init__(self, n_class, use_patch_enc, enc_type, pretrained, n_chan_in, n_res_blocks,
        use_pos, task_dict, N, M, I, D, H, D_k, D_v, D_inner, dropout, attn_dropout, device):
        super().__init__()

        self.M = M
        self.I = I
        self.D = D 
        self.use_pos = use_pos
        self.device = device

        if use_patch_enc:
            self.patch_encoder = self.get_patch_enc(enc_type, pretrained, n_chan_in, n_res_blocks))

        # define the multi-head cross-attention transformer
        self.transf = Transformer(n_token, H, D, D_k, D_v, D_inner, attn_dropout, dropout)

        if use_pos:
            self.pos_enc = pos_enc_1d(D, N).unsqueeze(0).to(device)
        
        # define output layer(s)
        self.output_layers = self.get_output_layers(task_dict)

    def score_and_select(self, emb, emb_pos, M, idx):
        # scores embeddings and selects the top-M embeddings
        B = emb.shape[0]
        D = self.D

        emb_to_score = emb_pos if torch.is_tensor(emb_pos) else emb

        attn = self.transf.get_scores(emb_to_score)

        top_idx = torch.topk(attn, M, dim = -1)[1]
        mem_emb = torch.gather(emb, 1, top_idx.unsqueeze(-1).repeat(1,1,D))

        mem_idx = torch.gather(idx, 1, top_idx.view(B, -1))
        return mem_emb, mem_idx

    def ips(self, patches):
        # get useful info
        M = self.M
        I = self.I
        D = self.D  
        device = self.device
        use_pos = self.use_pos
        patch_shape = patches.shape
        B, N = patch_shape[:2]

        # IPS required?
        if M >= N:
            return patches.to(device)      

        # should patch encoder be used?
        if len(patch_shape) == 3: # B, N, D
            is_image = False
        elif len(patches.shape) == 5: # B, N, n_chan_in, height, width
            is_image = True
        else:
            raise ValueError('The input is neither an image (5 dim) nor a feature vector (3 dim).')
        
        # IPS runs in no-gradient mode
        with torch.no_grad():
            # IPS runs in evaluation mode
            if self.training:
                self.patch_encoder.eval()
                self.transf.eval()
            
            # adjust positional encoding to batch
            if use_pos:
                pos_enc = self.pos_enc.repeat(B, 1, 1)

            # init memory
            init_patch = patches[:,:M].to(device)
            if is_image:
                mem_emb = self.patch_encoder(init_patch.reshape(-1, *patch_shape.shape[2:]))
                mem_emb = mem_emb.view(B, M, -1)
            else:
                mem_emb = init_patch
            
            # init mem idx
            idx = torch.arange(N, dtype=torch.int64, device=device).unsqueeze(0).repeat(B, 1)
            mem_idx = idx[:,:M]

            # get num iterations
            n_iter = math.ceil((N - M) / I)
            for i in range(n_iter):
                # get next patches
                start_idx = i * I + M
                end_idx = min(start_idx + I, N)

                iter_patch = patches[:, start_idx:end_idx].to(device)
                iter_idx = idx[:, start_idx:end_idx]
                if use_pos:
                    iter_pos_enc = pos_enc[:, start_idx:end_idx]

                # embed patches
                if is_image:
                    iter_emb = self.patch_encoder(iter_patch.reshape(-1, *patch_shape.shape[2:]))
                    iter_emb = iter_emb.view(B, -1, D)
                else:
                    iter_emb = iter_patch
                
                # concatenate with memory buffer
                all_emb = torch.cat(mem_emb, iter_emb)
                all_idx = torch.cat((mem_idx, iter_idx), dim=-1)
                if use_pos:
                    all_emb_pos = iter_emb + iter_pos_enc
                else:
                    all_emb_pos = None

                mem_emb, mem_idx = self.score_and_select(all_emb, all_emb_pos, M, all_idx)
            
            # select patches
            mem_patch = torch.gather(patches, 1, 
                mem_idx.view(B, -1, torch.ones(patch_shape[2:])).repeat(1, 1, *patch_shape[2:]).to(patches.device)
            )
            if use_pos:
                mem_pos = torch.gather(pos_enc, 1, mem_idx.unsqueeze(-1).repeat(1, 1, D))
            else:
                mem_pos = None

            # set network back to gradient and training mode            
            if self.training:
                self.patch_encoder.train()
                self.transf.train()
    
    return mem_patch.to(device), mem_pos

    def forward(self, mem_patch, mem_pos):
        patch_shape = mem_patch.shape
        B, M = patch_shape.shape[:2]

        if len(patch_shape) == 3: # B, N, D
            is_image = False
        elif len(patches.shape) == 5: # B, N, n_chan_in, height, width
            is_image = True
        else:
            raise ValueError('The input is neither an image (5 dim) nor a feature vector (3 dim).')
        
        if is_image:
            mem_emb = self.patch_encoder(mem_patch.reshape(-1, *patch_shape.shape[2:]))
            mem_emb = mem_emb.view(B, M, -1)        

        if torch.is_tensor(mem_pos):
            mem_emb = mem_emb + mem_pos

        image_emb = self.transf(mem_emb, return_attns=False)

        preds = []
        for i, layer in enumerate(self.outp_layers):
            pred = layer(image_emb[:,i]) # .view(b, -1)
            preds.append(pred)
 
        return preds