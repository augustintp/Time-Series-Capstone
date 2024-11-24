import os
from datetime import datetime
import pytz

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

from utils.metrics import metric
from torch.optim.lr_scheduler import _LRScheduler

plt.switch_backend('agg')


def name_with_datetime():
    now = datetime.now(tz=pytz.utc)
    now = now.astimezone(pytz.timezone('US/Pacific'))
    return now.strftime("%Y-%m-%d_%H:%M:%S")


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


class CosAnnealWarmupRestarts(_LRScheduler):
    """
        Reference: https://github.com/microsoft/LoRA.

        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.0.
        warmup_steps(int): Linear warmup step size. Default: 0.
        alpha (float): Decrease rate of max learning rate by cycle. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_lr: float = 0.1,
            min_lr: float = 0.0,
            warmup_steps: int = 0,
            max_steps: int = 1,
            alpha: float = 0.,
            last_epoch: int = -1
    ):
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size

        self.alpha = alpha  # decrease rate of max learning rate by cycle
        self.max_steps = max_steps
        super(CosAnnealWarmupRestarts, self).__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            curr_lr = self.max_lr * self.last_epoch / self.warmup_steps
            return curr_lr
        else:
            _step = min(self.last_epoch, self.max_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * _step / self.max_steps))
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            return self.max_lr * decayed  # learning_rate * decayed

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = math.floor(epoch)
        _lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = _lr


def plot_by_time_group(global_step, setting, folder_path, merged_df, all_pred_cols, tgt_cols,
                       plot_title, writer=None):
    row_titles = ['day', 'month']
    col_titles = ['mae', 'mse']
    col_colors = ['b', 'g', 'r', 'c', 'm']
    nrows = len(row_titles)
    ncols = len(col_titles)
    row_range = {'day': range(7), 'month': range(1, 13)}
    group_metrics = {row_name: {col_name: [] for col_name in col_titles} for row_name in row_titles}

    for row_title in row_titles:
        for metrics_time in row_range[row_title]:
            if row_title == 'day':
                time_df = merged_df[merged_df['date'].dt.dayofweek == metrics_time]
            elif row_title == 'month':
                time_df = merged_df[merged_df['date'].dt.month == metrics_time]
            else:
                raise ValueError(f'Unknown row_title: {row_title}')

            if time_df.empty:
                mae, mse = 0, 0
            else:
                mae, mse, _, _, _ = metric(time_df[all_pred_cols].to_numpy(),
                                           time_df[tgt_cols].to_numpy())
            group_metrics[row_title]['mae'].append(mae)
            group_metrics[row_title]['mse'].append(mse)
            if writer is not None:
                writer.add_scalar(f"{row_title}/{metrics_time}_mae", mae, global_step)
                writer.add_scalar(f"{row_title}/{metrics_time}_mse", mse, global_step)

    f = plt.figure(figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
    f.suptitle(f'Step {global_step} - {setting}')
    ax = np.array(f.subplots(nrows, ncols)).reshape(nrows, ncols)  # Ensure ax is always 2D

    for j, row_title in enumerate(row_titles):
        for k, col_title in enumerate(col_titles):
            ax[j, k].bar(row_range[row_title], group_metrics[row_title][col_title], color=col_colors[k])
            ax[j, k].set_title(f'{row_title} - {col_title}')
    f.savefig(os.path.join(folder_path, 'figures', f'{plot_title}_time_metrics.png'), bbox_inches='tight')
    plt.close()
