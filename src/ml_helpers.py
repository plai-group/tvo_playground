from __future__ import division
import sys
import os
import torch
from torch._six import inf
import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed
import joblib
from pathlib import Path

persist_dir = Path('./.persistdir')


class AverageMeter(object):
    """
    Computes and stores the average, var, and sample_var
    Taken from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, name="AverageMeter", fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.count = 0
        self.M2 = 0

        self.mean = 0
        self.variance = 0
        self.sample_variance = 0

    def step(self, val):
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.M2 += delta * delta2

        self.variance = self.M2 / self.count if self.count > 2 else 0
        self.sample_variance = self.M2 / \
            (self.count - 1) if self.count > 2 else 0

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({var' + self.fmt + '})'
        return fmtstr.format(name=self.name, val=self.mean, var=self.variance)


class MovingAverageMeter(object):
    """Computes the  moving average of a given float."""

    def __init__(self, name, fmt=':f', window=5):
        self.name = "{} (window = {})".format(name, window)
        self.fmt = fmt
        self.N = window
        self.history = []
        self.val = None
        self.reset()

    def reset(self):
        self.val = None
        self.history = []

    def step(self, val):
        self.history.append(val)
        self.previous = self.val
        if self.val is None:
            self.val = val
        else:
            window = self.history[-self.N:]
            self.val = sum(window) / len(window)
            if len(window) == self.N:
                self.history == window
        return self.val

    @property
    def relative_change(self):
        if None not in [self.val, self.previous]:
            relative_change = (self.previous - self.val) / self.previous
            return relative_change
        else:
            return 0

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(name=self.name, val=self.val, avg=self.relative_change)


class ConvergenceMeter(object):
    """This is a modification of pytorch's ReduceLROnPlateau object
        (https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau)
        which acts as a convergence meter. Everything
        is the same as ReduceLROnPlateau, except it doesn't
        require an optimizer and doesn't modify the learning rate.
        When meter.converged(loss) is called it returns a boolean that
        says if the loss has converged.

    Args:
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity metered has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity metered has stopped increasing. Default: 'min'.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> meter = Meter('min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     if meter.converged(val_loss):
        >>>         break
    """

    def __init__(self, mode='min', patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, eps=1e-8):

        self.has_converged = False
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.has_converged = True

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' +
                             threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode


class BestMeter(object):
    """ This is like ConvergenceMeter except it stores the
        best result in a set of results. To be used in a
        grid search

    Args:
        mode (str): One of `min`, `max`. In `min` mode, best will
            be updated when the quantity metered is lower than the current best;
            in `max` mode best will be updated when the quantity metered is higher
            than the current best. Default: 'max'.

    """

    def __init__(self, mode='max', verbose=True):

        self.has_converged = False
        self.verbose = verbose
        self.mode = mode
        self.best = None
        self.best_obj = None
        self.mode_worse = None  # the worse value for the chosen mode
        self._init_is_better(mode=mode)
        self._reset()

    def _reset(self):
        self.best = self.mode_worse

    def step(self, metrics, best_obj=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if self.is_better(current, self.best):
            if self.verbose:
                print("*********New best**********")
                print("value: ", current)
                print("object: ", best_obj)
                print("***************************")
            self.best = current
            self.best_obj = best_obj
            return True

        return False

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best

    def _init_is_better(self, mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode


# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__


def get_data_loader(dataset, batch_size, args, shuffle=True):
    """Args:
        np_array: shape [num_data, data_dim]
        batch_size: int
        device: torch.device object

    Returns: torch.utils.data.DataLoader object
    """

    if args.device == torch.device('cpu'):
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}

    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def split_train_test_by_percentage(dataset, train_percentage=0.8):
    """ split pytorch Dataset object by percentage """
    train_length = int(len(dataset) * train_percentage)
    return torch.utils.data.random_split(dataset, (train_length, len(dataset) - train_length))


def pmap(f, arr, n_jobs=-1, prefer='threads', verbose=10):
    return Parallel(n_jobs=n_jobs, prefer=prefer, verbose=verbose)(delayed(f)(i) for i in arr)


def put(value, filename):
    persist_dir.mkdir(exist_ok=True)
    filename = persist_dir / filename
    print("Saving to ", filename)
    joblib.dump(value, filename)


def get(filename):
    filename = persist_dir / filename
    assert filename.exists(), "{} doesn't exist".format(filename)
    print("Saving to ", filename)
    return joblib.load(filename)


def smooth(arr, window):
    return pd.Series(arr).rolling(window, min_periods=1).mean().values


def tensor(data, args=None, dtype=torch.float):
    device = torch.device('cpu') if args is None else args.device
    if torch.is_tensor(data):
        return data.to(dtype=dtype, device=device)
    else:
        return torch.tensor(np.array(data), device=device, dtype=dtype)


def is_test_time(epoch, args):
    if args.train_only:
        return False

    # last epoch
    if epoch == (args.epochs - 1):
        return True

    # test epoch
    if (args.test_during_training and ((epoch % args.test_frequency) == 0)):
        return True

    # Else
    return False


def detect_cuda(args):
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False
    return args


def is_schedule_update_time(epoch, args):
    # No scheduling
    if args.loss != 'tvo':
        return False

    # First epoch, initalize
    if epoch == 0:
        return True

    # Update happens at each minibatch
    if args.per_sample is True:
        return False

    # Initalize once and never update
    if args.schedule_update_frequency == 0:
        return False

    # catch checkpoint epoch
    if (epoch % args.schedule_update_frequency) == 0:
        return True

    # Else
    return False


def is_checkpoint_time(epoch, args):
    # No checkpointing
    if args.checkpoint is False:
        return False

    # skip first epoch
    if (epoch == 0):
        return False

    # catch last epoch
    if epoch == (args.epochs - 1):
        return True

    # catch checkpoint epoch
    if (epoch % args.checkpoint_frequency) == 0:
        return True

    # Else
    return False


def is_gradient_time(epoch, args):
    # No checkpointing
    if args.save_grads is False:
        return False

    # catch checkpoint epoch
    if (epoch % args.test_frequency) == 0:
        return True

    # Else
    return False


def logaddexp(a, b):
    """Returns log(exp(a) + exp(b))."""

    return torch.logsumexp(torch.cat([a.unsqueeze(0), b.unsqueeze(0)]), dim=0)


def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a tensor.
    """

    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    # log_numerator = values
    return values - log_denominator


def make_sparse(sparse_mx, args):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    indices = tensor(
        np.vstack((sparse_mx.row, sparse_mx.col)), args, torch.long)
    values = tensor(sparse_mx.data, args)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    return torch.exp(lognormexp(values, dim=dim))


def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_grads(model):
    return torch.cat([torch.flatten(p.grad.clone()) for p in model.parameters()]).cpu()


def log_ess(log_weight):
    """Log of Effective sample size.
    Args:
        log_weight: Unnormalized log weights
            torch.Tensor [batch_size, S] (or [S])
    Returns: log of effective sample size [batch_size] (or [1])
    """
    dim = 1 if log_weight.ndimension() == 2 else 0

    return 2 * torch.logsumexp(log_weight, dim=dim) - \
        torch.logsumexp(2 * log_weight, dim=dim)


def ess(log_weight):
    """Effective sample size.
    Args:
        log_weight: Unnormalized log weights
            torch.Tensor [batch_size, S] (or [S])
    Returns: effective sample size [batch_size] (or [1])
    """

    return torch.exp(log_ess(log_weight))
