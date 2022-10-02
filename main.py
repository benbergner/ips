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

dataset = 'mnist'

with open(os.path.join('config', dataset + '_config.yml'), "r") as ymlfile:
    cfg = yaml.load(ymlfile)

data_dir = cfg['dset']['data_dir']
B = cfg['opt']['B']
B_seq = cfg['opt']['B_seq']

train_data = MegapixelMNIST(data_dir=data_dir, patch_size=patch_size, patch_stride=patch_stride, train=True)
test_data = MegapixelMNIST(data_dir=data_dir, patch_size=patch_size, patch_stride=patch_stride, train=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=B_seq, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=B_seq, shuffle=False, num_workers=num_workers, pin_memory=True)