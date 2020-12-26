
import numpy as np
import torch
import torch.nn as nn
from math import pi as pi
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.autograd import Variable



def singleton_repeat(x, n, sheldon_repeat=True):
    """ 
    Repeat a batch of data n times. 
    It's the safe way to repeat
    First add an additional dimension, repeat that dimention, then reshape it back. 
    So that later when reshaping, it's guranteed to follow the same tensor convention. 
     """
    if n == 1:
        return x
    else:
        # sheldon's DOES NOT WORK
        if sheldon_repeat:
            singleton_x = torch.unsqueeze(x, 0)
            repeated_x = singleton_x.repeat(n, 1, 1)
        else:
        #either of these should work
            #repeated_x =  x.repeat_interleave(n, dim=0)
            singleton_x = torch.unsqueeze(x, 1)
            repeated_x = singleton_x.repeat(1, n, 1)
        return repeated_x.view(-1, x.size()[-1])


def sum_except_batch(x, num_batch_dims= 1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    #if not check.is_nonnegative_int(num_batch_dims):
    #    raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def logaddexp(a, b):
    return torch.logsumexp(torch.cat([a.unsqueeze(0), b.unsqueeze(0)], dim=0), dim =0)

def lognormexp(values, dim=-1):
    """Exponentiates, normalizes and takes log of a tensor.
    """

    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    return values - log_denominator

def exponentiate_and_normalize(values, dim = -1):
    return torch.exp(lognormexp(values, dim=dim))


def log_sum_exp(x, dim=1):
    """ based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    """
    return torch.logsumexp(x, dim = dim)
    #max_, _ = torch.max(x, dim, keepdim=True, out=None)
    #return torch.log(torch.sum(torch.exp(x - max_), dim)) + torch.squeeze(max_)


def log_mean_exp(x, dim=1):
    """ based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    """
    #return torch.logsumexp(x, dim = dim) - torch.log(x.shape[dim]* torch.mean(torch.ones_like(x)))
    max_, _ = torch.max(x, dim, keepdim=True, out=None)
    return torch.log(torch.mean(torch.exp(x - max_), dim)) + torch.squeeze(max_)



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
