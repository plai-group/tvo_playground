import torch
from functools import partial
import numpy as np
from src.utils.record_utils import calc_exp



def get_partition(args):
    """Create a non-decreasing sequence of values between zero and one.
    See https://en.wikipedia.org/wiki/Partition_of_an_interval.

    Args:
        args.num_partitions: length of sequence minus one
        args.schedule: \'linear\' or \'log\'
        args.log_beta_min: log (base ten) of beta_min. only used if partition_type
            is log. default -10 (i.e. beta_min = 1e-10).
        args.device: torch.device object (cpu by default)

    Returns: tensor of shape [num_partitions + 1]
    """
    if args.K == 1:
        partition = tensor((0., 1), args)
    else:
        if args.schedule == 'linear':
            partition = torch.linspace(0, 1, steps=args.K + 1,
                                       device=args.device)
        elif args.schedule == 'log':
            partition = torch.zeros(args.K + 1, device=args.device,
                                    dtype=torch.float)
            partition[1:] = torch.logspace(args.log_beta_min, 0, steps=args.K, device=args.device,
                                           dtype=torch.float)
        else:
            # DEFAULT IS TO START WITH LOG
            partition = torch.zeros(args.K + 1, device=args.device,
                                    dtype=torch.float)
            partition[1:] = torch.logspace(args.log_beta_min, 0, steps=args.K, device=args.device,
                                           dtype=torch.float)
    return partition

def get_partition_scheduler(args):
    """
    Args:
        args : arguments from main.py
    Returns:
        callable beta_update function
    * callable has interface f(log_iw, args, **kwargs)
    * returns beta_id, or unchanged args.partition, by default
    Beta_update functions:
        *** MUST be manually specified here
        * given via args.partition_type and other method-specific args
        * should handle 0/1 endpoint insertion internally
        * may take args.K - 1 partitions as a result (0 is given)
    """

    schedule = args.schedule
    if schedule in ['log', 'linear']:
        return beta_id
    elif schedule == 'moments':
        return moments
    else:
        raise ValueError

def moments(model, args=None, **kwargs):
    args  = model.args if args is None else args
    start = 0
    stop  = 1
    threshold = 0.05

    if not args.per_sample and not args.per_batch:
        log_iw = model.get_total_log_weight(args, args.valid_S)
    else:
        log_iw = model.elbo()

    partitions = args.K-1
    targets = np.linspace(0.0, 1.0, num=args.K+1, endpoint=True)

    left  = calc_exp(log_iw, start, all_sample_mean= not(args.per_sample))
    right = calc_exp(log_iw, stop, all_sample_mean= not(args.per_sample))
    left  = torch.mean(left, axis = 0, keepdims = True) if args.per_batch else left
    right = torch.mean(right, axis = 0, keepdims= True) if args.per_batch else right
    moment_avg = right - left

    beta_result = []
    for t in range(len(targets)):
        if targets[t] == 0.0 or targets[t] == 1.0:
            beta_result.append(targets[t] * (torch.ones_like(log_iw[:,0]) if args.per_sample else 1) ) # zero if targets[t]=0
        else:
            target = targets[t]
            moment = left + target*moment_avg #for t in targets]

            start = torch.zeros_like(log_iw[:,0]) if args.per_sample else torch.zeros_like(left)
            stop = torch.ones_like(log_iw[:,0]) if args.per_sample else torch.ones_like(left)

            beta_result.append(_moment_binary_search(\
                    moment, log_iw, start = start, stop = stop, \
                        threshold=threshold, per_sample = args.per_sample))

    if args.per_sample: #or args.per_batch:
        beta_result = torch.cat([b.unsqueeze(1) for b in beta_result], axis=1).unsqueeze(1)
        beta_result, _ = torch.sort(beta_result, -1)
    else:
        beta_result = torch.cuda.FloatTensor(beta_result)

    return beta_result

def _moment_binary_search(target, log_iw, start=0, stop= 1, threshold = 0.1, recursion = 0, per_sample = False, min_beta = 0.001): #recursion = 0,
    beta_guess = .5*(stop+start)
    eta_guess = calc_exp(log_iw, beta_guess, all_sample_mean = not per_sample).squeeze()
    target = torch.ones_like(eta_guess)*(target.squeeze())
    start_ = torch.where( eta_guess <  target,  beta_guess, start)
    stop_ = torch.where( eta_guess >  target, beta_guess , stop)

    if torch.sum(  torch.abs( eta_guess - target) > threshold ).item() == 0:
        return beta_guess
    else:
        if recursion > 500:
            return beta_guess
        else:
            return _moment_binary_search(
                target,
                log_iw,
                start= start_,
                stop= stop_,
                recursion = recursion + 1,
                per_sample = per_sample)

def beta_id(model, args = None, **kwargs):
    """
    dummy beta update for static / unspecified partition_types
    """
    return args.partition


def get_tvo_multipliers(partition, integration = 'left'):
    """ Outputs partition multipers depending on integration rule
     Args:
         partition = partition of interval [0,1]
         integration : left, right, trapz, single (i.e. 1 * partition[*, 1])

        (helper function to accomodate per_sample calculations)

     Returns: tensor with size = partition.shape
     """
    integration = 'left' if integration is None else integration
    if len(partition.shape) == 1:
        multiplier = torch.zeros_like(partition)
        if integration == 'trap':
            multiplier[0] = 0.5 * (partition[1] - partition[0])
            multiplier[1:-1] = 0.5 * (partition[2:] - partition[0:-2])
            multiplier[-1] = 0.5 * (partition[-1] - partition[-2])
        elif integration == 'left':
            multiplier[:-1] = partition[1:] - partition[:-1]
        elif integration == 'right':
            multiplier[1:] = partition[1:] - partition[:-1]

    else:
        multiplier = torch.zeros_like(partition)
        if integration == 'trap':
            multiplier[..., 0] = 0.5 * (partition[..., 1] - partition[..., 0])
            multiplier[..., 1:-1] = 0.5 * (partition[..., 2:] - partition[..., 0:-2])
            multiplier[..., -1] = 0.5 * (partition[..., -1] - partition[..., -2])
        elif integration == 'left':
            multiplier[..., :-1] = partition[..., 1:] - partition[..., :-1]
        elif integration == 'right':
            multiplier[..., 1:] = partition[..., 1:] - partition[..., :-1]
        elif integration == 'single':
            multiplier = torch.ones_like(partition)
            if multiplier.shape[-1] == 3:
                multiplier[..., 0] = 0
                multiplier[..., -1] = 0

    return multiplier
