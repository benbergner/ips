#!/usr/bin/env python

import os
import sys
import yaml

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import adjust_learning_rate, eps, Evaluator
from data.megapixel_mnist.mnist_dataset import MegapixelMNIST
from architecture.ips_net import IPSNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = 'mnist' # either one of {'mnist', 'camelyon', 'traffic'}

# get config
with open(os.path.join('config', dataset + '_config.yml'), "r") as ymlfile:
    c = yaml.load(ymlfile, Loader=yaml.FullLoader)

    n_epoch, n_epoch_warmup, B, B_seq, lr, wd = c['n_epoch'], c['n_epoch_warmup'], c['B'], c['B_seq'], c['lr'], c['wd']
    n_class, data_dir, n_worker = c['n_class'], c['data_dir'], c['n_worker']
    use_patch_enc, enc_type, pretrained, n_chan_in, n_res_blocks = c['use_patch_enc'], c['enc_type'], c['pretrained'], c['n_chan_in'], c['n_res_blocks']
    n_token, N, M, I, patch_size, patch_stride = c['n_token'], c['N'], c['M'], c['I'], c['patch_size'], c['patch_stride']
    use_pos, H, D, D_k, D_v, D_inner, attn_dropout, dropout = c['use_pos'], c['H'], c['D'], c['D_k'], c['D_v'], c['D_inner'], c['attn_dropout'], c['dropout']
    task_dict = c['tasks']

# define datasets and dataloaders
train_data = MegapixelMNIST(data_dir, patch_size, patch_stride, task_dict, train=True)
test_data = MegapixelMNIST(data_dir, patch_size, patch_stride, task_dict, train=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=B_seq, shuffle=True, num_workers=n_worker, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=B_seq, shuffle=False, num_workers=n_worker, pin_memory=True)

# define network
net = IPSNet(n_class, use_patch_enc, enc_type, pretrained, n_chan_in, n_res_blocks, use_pos, task_dict,
    n_token, N, M, I, D, H, D_k, D_v, D_inner, dropout, attn_dropout, device
).to(device)

loss_nll = nn.NLLLoss()
loss_bce = nn.BCELoss()

optimizer = torch.optim.AdamW(net.parameters(), lr=0, weight_decay=wd)

loss_fns = {}
for task in task_dict.values():
    loss_fns[task['name']] = loss_nll if task['act_fn'] == 'softmax' else loss_bce

train_evaluator = Evaluator(task_dict)
test_evaluator = Evaluator(task_dict)

for epoch in range(n_epoch):
    
    # Training
    net.train()

    n_prep, n_prep_total = 0, 0
    start_new_batch = True

    for data_it, data in enumerate(train_loader, start=epoch * len(train_loader)):
        #patches_iter, labels_iter, max_labels_iter, top_labels_iter, multi_labels_iter = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device).float()
        image_patches = data['input'].to(device)

        if start_new_batch:
            mem_patch = torch.zeros((B, M, n_chan_in, *patch_size)).to(device)
            if use_pos:
                mem_pos_enc = torch.zeros((B, M, D)).to(device)

            labels = {}
            for task in task_dict.values():
                if task['multi_label']:
                    labels[task['name']] = torch.zeros((B, n_class), dtype=torch.float32).to(device)
                else:
                    labels[task['name']] = torch.zeros((B,), dtype=torch.int64).to(device)
            
            start_new_batch = False
        
        mem_patch_iter, mem_pos_enc_iter = net.ips(image_patches)
        
        # fill batch
        n_seq, len_seq = mem_patch_iter.shape[:2]
        mem_patch[n_prep:n_prep+n_seq, :len_seq] = mem_patch_iter
        if use_pos:
            mem_pos_enc[n_prep:n_prep+n_seq, :len_seq] = mem_pos_enc_iter
        
        for task in task_dict.values():
            labels[task['name']][n_prep:n_prep+n_seq] = data[task['name']]
        
        n_prep += n_seq
        n_prep_total += n_seq

        batch_full = (n_prep == B)
        is_last_batch = n_prep_total == len(train_loader)

        if batch_full or is_last_batch:
            if not batch_full:
                # shrink batch
                mem_patch = mem_patch[:n_prep]
                if use_pos:
                    mem_pos_enc = mem_pos_enc[:n_prep]
                
                for task in task_dict.values():
                    labels[task['name']] = labels[task['name']][:n_prep]
            
            adjust_learning_rate(n_epoch_warmup, n_epoch, lr, optimizer, train_loader, data_it+1)
            optimizer.zero_grad()

            preds = net(mem_patch, mem_pos_enc)

            loss = 0
            task_losses, task_preds, task_labels = {}, {}, {}
            for task in task_dict.values():
                t = task['name']

                loss_fn = loss_fns[t]
                label = labels[t]
                pred = preds[t]

                if task['act_fn'] == 'softmax':
                    pred = torch.log(pred + eps)

                if task['multi_label']:
                    pred = pred.view(-1)
                    label = label.view(-1)

                task_preds[t] = pred.cpu().numpy()
                task_labels[t] = label.cpu().numpy()
                task_loss = loss_fn(pred, label)
                task_losses[t] = task_loss.item()

                loss += task_loss
                
            loss /= len(task_dict.values())

            loss.backward()
            optimizer.step()

            train_evaluator.update(task_losses, task_preds, task_labels)

            n_prep = 0
            start_new_batch = True
    
    train_evaluator.compute_metric()
    train_evaluator.print_stats(epoch)

    # Evaluation
    n_prep, n_prep_total = 0, 0
    start_new_batch = True

    net.eval()
    with torch.no_grad():
        for data in test_loader:
            image_patches = data['input'].to(device)

            if start_new_batch:
                mem_patch = torch.zeros((B, M, n_chan_in, *patch_size)).to(device)
                if use_pos:
                    mem_pos_enc = torch.zeros((B, M, D)).to(device)

                labels = {}
                for task in task_dict.values():
                    if task['multi_label']:
                        labels[task['name']] = torch.zeros((B, n_class), dtype=torch.float32).to(device)
                    else:
                        labels[task['name']] = torch.zeros((B,), dtype=torch.int64).to(device)
                
                start_new_batch = False
            
            mem_patch_iter, mem_pos_enc_iter = net.ips(image_patches)
        
            n_seq, len_seq = mem_patch_iter.shape[:2]
            mem_patch[n_prep:n_prep+n_seq, :len_seq] = mem_patch_iter
            if use_pos:
                mem_pos_enc[n_prep:n_prep+n_seq, :len_seq] = mem_pos_enc_iter
            
            for task in task_dict.values():
                labels[task['name']][n_prep:n_prep+n_seq] = data[task['name']]
            
            n_prep += n_seq
            n_prep_total += n_seq

            batch_full = (n_prep == B)
            is_last_batch = n_prep_total == len(test_loader)

            if batch_full or is_last_batch:
                if not batch_full:
                    mem_patch = mem_patch[:n_prep]
                    if use_pos:
                        mem_pos_enc = mem_pos_enc[:n_prep]
                    
                    for task in task_dict.values():
                        labels[task['name']] = labels[task['name']][:n_prep]
                
                preds = net(mem_patch, mem_pos_enc)

                loss = 0
                task_losses, task_preds, task_labels = {}, {}, {}
                for task in task_dict.values():
                    t = task['name']

                    loss_fn = loss_fns[t]
                    label = labels[t]
                    pred = preds[t]

                    if task['act_fn'] == 'softmax':
                        pred = torch.log(pred + eps)

                    if task['multi_label']:
                        pred = pred.view(-1)
                        label = label.view(-1)
                    
                    task_preds[t] = pred.cpu().numpy()
                    task_labels[t] = label.cpu().numpy()
                    task_loss = loss_fn(pred, label)
                    task_losses[t] = task_loss.item()

                    loss += task_loss

                loss /= len(task_dict.values())

                n_prep = 0
                start_new_batch = True
    
    test_evaluator.compute_metric()
    test_evaluator.print_stats(epoch)