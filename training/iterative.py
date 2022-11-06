import sys
import numpy as np
import torch

from utils.utils import adjust_learning_rate

def init_batch(device, conf):

    mem_patch = torch.zeros((conf.B, conf.M, conf.n_chan_in, *conf.patch_size)).to(device)
    if conf.use_pos:
        mem_pos_enc = torch.zeros((conf.B, conf.M, conf.D)).to(device)
    else:
        mem_pos_enc = None

    labels = {}
    for task in conf.tasks.values():
        if task['multi_label']:
            labels[task['name']] = torch.zeros((conf.B, conf.n_class), dtype=torch.float32).to(device)
        else:
            labels[task['name']] = torch.zeros((conf.B,), dtype=torch.int64).to(device)
    
    return mem_patch, mem_pos_enc, labels

def fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
            mem_patch_iter, mem_pos_enc_iter, conf):

    n_seq, len_seq = mem_patch_iter.shape[:2]
    mem_patch[n_prep:n_prep+n_seq, :len_seq] = mem_patch_iter
    if conf.use_pos:
        mem_pos_enc[n_prep:n_prep+n_seq, :len_seq] = mem_pos_enc_iter
    
    for task in conf.tasks.values():
        labels[task['name']][n_prep:n_prep+n_seq] = data[task['name']]
    
    n_prep += n_seq
    n_prep_batch += 1

    batch_data = (mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch)

    return batch_data

def shrink_batch(mem_patch, mem_pos_enc, labels, n_prep, conf):
    mem_patch = mem_patch[:n_prep]
    if conf.use_pos:
        mem_pos_enc = mem_pos_enc[:n_prep]
    
    for task in conf.tasks.values():
        labels[task['name']] = labels[task['name']][:n_prep]
    
    return mem_patch, mem_pos_enc, labels

def compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf):

    preds = net(mem_patch, mem_pos_enc)

    loss = 0
    task_losses, task_preds, task_labels = {}, {}, {}
    for task in conf.tasks.values():
        t_name, t_act, t_multi = task['name'], task['act_fn'], task['multi_label']

        criterion = criterions[t_name]
        label = labels[t_name]
        pred = preds[t_name]

        if t_act == 'softmax':
            pred_loss = torch.log(pred + conf.eps)
        else:
            pred_loss = pred

        if t_multi:
            pred_loss = pred_loss.view(-1)
            label_loss = label.view(-1)
        else:
            label_loss = label

        task_loss = criterion(pred_loss, label_loss)
        task_losses[t_name] = task_loss.item()
        task_preds[t_name] = pred.detach().cpu().numpy()
        task_labels[t_name] = label.detach().cpu().numpy()

        loss += task_loss
        
    loss /= len(conf.tasks.values())

    return loss, [task_losses, task_preds, task_labels]

def train_one_epoch(net, criterions, data_loader, optimizer, device, epoch, log_writer, conf):
    
    net.train()

    n_prep, n_prep_batch = 0, 0
    mem_pos_enc = None
    start_new_batch = True

    times = []
    for data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        image_patches = data['input'].to(device) if conf.eager else data['input']

        # create buffer for selected patches
        if start_new_batch:
            if conf.track_efficiency:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            mem_patch, mem_pos_enc, labels = init_batch(device, conf)
            start_new_batch = False
        
        # apply IPS
        mem_patch_iter, mem_pos_enc_iter = net.ips(image_patches)
        
        # fill batch
        batch_data = fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
                        mem_patch_iter, mem_pos_enc_iter, conf)
        mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch = batch_data

        batch_full = (n_prep == conf.B)
        is_last_batch = n_prep_batch == len(data_loader)

        # train as soon as batch is full
        if batch_full or is_last_batch:

            if not batch_full:
                mem_patch, mem_pos_enc, labels = shrink_batch(mem_patch, mem_pos_enc, labels, n_prep, conf)
            
            adjust_learning_rate(conf.n_epoch_warmup, conf.n_epoch, conf.lr, optimizer, data_loader, data_it+1)
            optimizer.zero_grad()

            loss, task_info = compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf)
            task_losses, task_preds, task_labels = task_info

            loss.backward()
            optimizer.step()

            if conf.track_efficiency:
                end_event.record()
                torch.cuda.synchronize()
                if epoch == conf.track_epoch and data_it > 0 and not is_last_batch:
                    times.append(start_event.elapsed_time(end_event))
                    print("times: ", times[-1])

            log_writer.update(task_losses, task_preds, task_labels)

            n_prep = 0
            start_new_batch = True
    if conf.track_efficiency:
        if epoch == conf.track_epoch:
            print("avg. time: ", np.mean(times))

            stats = torch.cuda.memory_stats()
            peak_bytes_requirement = stats["allocated_bytes.all.peak"]
            print(f"Peak memory requirement: {peak_bytes_requirement / 1024 ** 3:.4f} GB")

            print("TORCH.CUDA.MEMORY_SUMMARY: ", torch.cuda.memory_summary())
            sys.exit()


@torch.no_grad()
def evaluate(net, criterions, data_loader, device, epoch, log_writer, conf):

    n_prep, n_prep_batch = 0, 0
    mem_pos_enc = None
    start_new_batch = True

    net.eval()
    
    for data in data_loader:
        image_patches = data['input'].to(device) if conf.eager else data['input']

        if start_new_batch:
            mem_patch, mem_pos_enc, labels = init_batch(device, conf)
            start_new_batch = False
        
        mem_patch_iter, mem_pos_enc_iter = net.ips(image_patches)
        
        batch_data = fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
                        mem_patch_iter, mem_pos_enc_iter, conf)
        mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch = batch_data

        batch_full = (n_prep == conf.B)
        is_last_batch = n_prep_batch == len(data_loader)

        if batch_full or is_last_batch:

            if not batch_full:
                mem_patch, mem_pos_enc, labels = shrink_batch(mem_patch, mem_pos_enc, labels, n_prep, conf)
            
            _, task_info = compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf)
            task_losses, task_preds, task_labels = task_info

            log_writer.update(task_losses, task_preds, task_labels)

            n_prep = 0
            start_new_batch = True