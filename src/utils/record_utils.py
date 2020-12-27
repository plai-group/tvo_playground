import torch
import numpy as np
from src.handlers.ml_helpers import AverageMeter, get_grads, tensor
from src.utils.math_utils import exponentiate_and_normalize
#lognormexp, exponentiate_and_normalize, seed_all
from torch.distributions.multinomial import Multinomial




def calculate_grad_variance(model, args):
    grad_var = AverageMeter()
    batch = next(iter(args.train_data_loader))
    for _ in range(10):
        model.zero_grad()
        loss, logpx, test_elbo = model.forward(batch)
        loss.backward()
        model_grads = get_grads(model)
        grad_var.step(model_grads)

    grad_std = grad_var.variance.sqrt().mean()
    return grad_std



def calc_exp(log_weight, args, all_sample_mean=True, snis=None):
    """
    Args:
        log_weight : [batch, samples, *]
        args : either args object or partition [batch, 1, K partitions]
        all_sample_mean : True averages over batch

    TO DO : replace for cleaner integration into code (pulled directly from Rob's)
    """
    log_weight = log_weight.unsqueeze(-1) if len(
        log_weight.shape) < 3 else log_weight
    if snis is None:
        try:  # args.partition or partition tensor directly
            partition = args.partition
        except:
            partition = args
        beta_iw = log_weight * partition
        snis = exponentiate_and_normalize(
            beta_iw, dim=1)
    else:
        pass

    exp = snis * log_weight
    exp = torch.sum(exp, dim=1)
    return torch.mean(exp, dim=0) if all_sample_mean else exp


def calc_var(log_weight,  args, snis=None, all_sample_mean=True):
    """
    Args:
        log_weight : [batch, samples, *]
        args : either args object or partition [batch, 1, K partitions]
        all_sample_mean : returns mean over samples if True
        snis : optionally feed weights to avoid recomputation
    Returns:
        Variance across importance samples at each beta (2nd derivative of logZβ)
    """
    log_weight = log_weight.unsqueeze(-1) if len(
        log_weight.shape) < 3 else log_weight

    if snis is None:
        try:  # args.partition or partition tensor directly
            partition = args.partition
        except:
            partition = args
        beta_iw = log_weight * partition
        snis = exponentiate_and_normalize(
            beta_iw, dim=1)
    else:
        pass

    exp_ = torch.sum(snis * log_weight, dim=1)
    exp2 = torch.sum(snis * torch.pow(log_weight, 2), dim=1)

    to_return = exp2 - torch.pow(exp_, 2)

    # VM: May have to switch to_return to E[(X-EX)(X-EX)] form, had numerical issues in the past
    assert not torch.isnan(to_return).any(), "Nan in calc_var() - switch to E[(X-EX)(X-EX)] form for numerical stability"

    return torch.mean(to_return, dim=0) if all_sample_mean else to_return


def calc_third(log_weight, args, snis=None, all_sample_mean=True):
    """
    Args:
        log_weight : [batch, samples, *]
        args : either args object or partition [batch, 1, K partitions]
        all_sample_mean : returns mean over samples if True
        snis : optionally feed weights to avoid recomputation
    Returns:
        Third derivative of logZβ at each beta
    """
    log_weight = log_weight.unsqueeze(-1) if len(
        log_weight.shape) < 3 else log_weight

    if snis is None:
        try:  # args.partition or partition tensor directly
            partition = args.partition
        except:
            partition = args
        beta_iw = log_weight * partition
        snis = exponentiate_and_normalize(
            beta_iw, dim=1)
    else:
        pass

    exp = torch.sum(snis * log_weight, dim=1)
    exp2 = torch.sum(snis * torch.pow(log_weight, 2), dim=1)
    var = exp2 - torch.pow(exp, 2)
    exp3 = torch.sum(snis * torch.pow(log_weight, 3), dim=1)

    to_return = exp3 - torch.pow(exp, 3) - 3*exp*var
    return torch.mean(to_return, dim=0) if all_sample_mean else to_return


def calc_fourth(log_weight, args, snis=None, all_sample_mean=True):
    """
    Args:
        log_weight : [batch, samples, *]
        args : either args object or partition [batch, 1, K partitions]
        all_sample_mean : returns mean over samples if True
        snis : optionally feed weights to avoid recomputation
    Returns:
        Fourth derivative of logZβ at each beta
    """
    log_weight = log_weight.unsqueeze(-1) if len(
        log_weight.shape) < 3 else log_weight

    if snis is None:
        try:  # args.partition or partition tensor directly
            partition = args.partition
        except:
            partition = args
        beta_iw = log_weight * partition
        snis = exponentiate_and_normalize(
            beta_iw, dim=1)
    else:
        pass

    exp = torch.sum(snis * log_weight, dim=1)
    exp2 = torch.sum(snis * torch.pow(log_weight, 2), dim=1)
    exp3 = torch.sum(snis * torch.pow(log_weight, 3), dim=1)
    exp4 = torch.sum(snis * torch.pow(log_weight, 4), dim=1)

    to_return = exp4 - 6*torch.pow(exp, 4) + 12*exp2 * \
        torch.pow(exp, 2) - 3*torch.pow(exp2, 2) - 4*exp*exp3
    return torch.mean(to_return, dim=0) if all_sample_mean else to_return