#!/usr/bin/env python

import os
import sys
import yaml

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import adjust_learning_rate
from data.megapixel_mnist.mnist_dataset import MegapixelMNIST
from architecture.ips_net import IPSNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = 'mnist' # either one of {'mnist', 'camelyon', 'traffic'}

# get config
with open(os.path.join('config', dataset + '_config.yml'), "r") as ymlfile:
    cfg = yaml.load(ymlfile)

# define datasets and dataloaders
train_data = MegapixelMNIST(data_dir=cfg['dset']['data_dir'], patch_size=cfg['ips']['patch_size'],
    patch_stride=cfg['ips']['patch_stride'], task_dict=cfg['tasks'], train=True)
test_data = MegapixelMNIST(data_dir=cfg['dset']['data_dir'], patch_size=cfg['ips']['patch_size'],
    patch_stride=cfg['ips']['patch_stride'], task_dict=cfg['tasks'], train=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg['opt']['B_seq'],
    shuffle=True, num_workers=cfg['dset']['num_workers'], pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg['opt']['B_seq'],
    shuffle=False, num_workers=cfg['dset']['num_workers'], pin_memory=True)

# define network
net = IPSNet(n_class=cfg['dset']['n_class'], use_patch_enc = cfg['enc']['use_patch_enc'],
    enc_type = cfg['enc']['enc_type'], pretrained = cfg['enc']['pretrained'],
    n_chan_in = cfg['enc']['n_chan_in'], n_res_blocks = cfg['enc']['n_res_blocks'],
    use_pos = cfg['aggr']['use_pos'], task_dict = cfg['tasks'], N = cfg['ips']['N'],
    M = cfg['ips']['M'], I = cfg['ips']['I'], D = cfg['aggr']['D'], H = cfg['aggr']['H'],
    D_k = cfg['aggr']['D_k'], D_v = cfg['aggr']['D_v'], D_inner = cfg['aggr']['D_inner'],
    dropout = cfg['aggr']['dropout'], attn_dropout = cfg['aggr']['attn_dropout'], device = device
).to(device)

loss_nll = nn.NLLLoss()
loss_bce = nn.BCELoss()

optimizer = torch.optim.AdamW(net.parameters(), lr=0, weight_decay=cfg['opt']['wd'])

loss_fns = {}
for task in cfg['tasks'].values():
    loss_fns[task['name']] = loss_nll if task['act_fn'] == 'softmax' else loss_bce

