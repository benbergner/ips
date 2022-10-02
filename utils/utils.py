import math

def adjust_learning_rate(n_epoch_warmup, n_epoch, max_lr, optimizer, dloader, step):
    max_steps = int(n_epoch * len(dloader))
    warmup_steps = int(n_epoch_warmup * len(dloader))
    if step < warmup_steps:
        lr = max_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = max_lr * 0.001
        lr = max_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr