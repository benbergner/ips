import math
import numpy as np
from sklearn.metrics import accuracy_score

#constants
eps = 1e-6

def adjust_learning_rate(n_epoch_warmup, n_epoch, max_lr, optimizer, dloader, step):
    """ adjust learning rate according to cosine schedule """

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

class Evaluator(nn.Module):
    ''' Stores and computes statistiscs of losses and metrics '''

    def __init__(self, task_dict):
        super().__init__()

        self.task_dict = task_dict
        self.losses_it = {}
        self.losses_epoch = {}
        self.y_preds = {}
        self.y_trues = {}
        self.metrics = {}

        for task in task_dict.values():
            t = task['name']
            losses_it[t] = []
            losses_epoch[t] = []
            y_preds[t] = []
            y_trues[t] = []
            metrics[t] = []
    
    def update(self, next_loss, next_y_pred, next_y_true):

        for task in self.task_dict.values():
            t = task['name']
            self.losses[t].append(next_loss[t])
            self.y_preds[t].extend(next_y_pred[t])
            self.y_trues[t].extend(next_y_true[t])

    def compute_metric(self):

        for task in self.task_dict.values():
            t = task['name']
            losses = self.losses_it[t]
            self.losses_epoch[t].append((np.mean(losses), np.std(losses)))

            current_metric = task['metric']
            if current_metric == 'accuracy':
                metric = accuracy_score(self.y_trues[t], self.y_preds[t])
                self.metrics[t].append(metric)
            
            # reset per iteration losses, preds and labels
            self.losses_it[t] = []
            self.y_preds[t] = []
            self.y_trues[t] = []


    def print_stats(self, epoch):
        print_str =  "Train Epoch: {} \n".format(epoch+1)
        for task in self.task_dict.values():
            t = task['name']
            metric_name = task['metric']
            mean_loss, std_loss = self.losses_epoch[t][epoch]
            metric = self.metrics[t][epoch]
            print_str += "task: {}, mean loss: {}, std loss: {}, {}: {}\n".format(t, mean_loss, std_loss, metric_name, metric)
