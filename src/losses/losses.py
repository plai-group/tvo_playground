import numpy as np
import torch
from src.utils import math_utils
from torch.nn import functional as F

def negative_bce(x_true, x_pred, x_logvar=None, dim = -1):
    """
    Args:
        x: [B,Z] : true data
        mean,logvar: [B,Z]   mean = predicted value / Bernoulli

    Returns:
        output: [B]
    """
    # l = nn.BCELoss(reduction='none')
    # output = l(x_pred, x_true)
    # return -torch.sum(output, dim = dim)

    if x_pred.shape == x_true.shape:
        return - torch.sum(F.binary_cross_entropy(x_pred, x_true, reduction='none'), dim=dim)
    else:
        while len(x_true.shape) < len(x_pred.shape):
            x_true = x_true.unsqueeze(1)

        for d in range(len(x_pred.shape)):
            if x_true.shape[d] < x_pred.shape[d]:
                x_true = torch.cat( [x_true]*int(x_pred.shape[d]/x_true.shape[d]) , dim = d)


        return - torch.sum(F.binary_cross_entropy(x_pred, x_true, reduction='none'), dim=dim)


def tvo_loss(snis_weights, integrand, args, partition = None):
    ''' 
    Haven't double checked, but written to expect
        -- SNIS / integrand tensors are [batch, S, K] or [k_per_chain, batch, S, K] (multi-sample)
    '''


    partition = partition if partition is not None else args.partition
    multiplier = get_tvo_multipliers (partition, args.integration)
    
    # sum over SNIS weighted samples
    expectation =  torch.sum(snis_weights * integrand, dim = -2)
    # sum over Δβ * E log p/q, mean over batch
    tvo = torch.mean( torch.sum( partition * expectation, dim=-1 ), dim =0)

    print()
    print("ATTEMPT AT CALCULATING TVO")
    import IPython
    IPython.embed()
    return -tvo


def iwae_loss(log_weights, num_chains=None, dim = None):
    if num_chains is None:
        dim = 0 if dim is None else dim
        return -torch.mean( math_utils.log_mean_exp(elbo, dim = dim) )
    else:
        iwae_reshape = elbo.view(( num_chains, -1))
        return -torch.mean( math_utils.log_mean_exp(iwae_reshape, dim = 0) )
        
        
def iwae_dreg_loss(elbo_with_stop_grad_q):

    normalized_weight = math_utils.exponentiate_and_normalize( elbo_with_stop_grad_q , dim=1)

    loss = torch.mean(torch.sum(torch.pow(normalized_weight,2).detach() * log_weight, 1), 0)
    return -loss


def evaluate_lower_bounds(model, data = None, S=100):
    with torch.no_grad():
        ''' TO DO : below is attempt to avoid re-calculating elbo, settle on one approach or clarify use cases? '''
        if model.internals.elbo is None or data is not None:
        	log_weight = model.elbo( data, S ) 
        else:
        	log_weight = model.internals.elbo

        iwae_bound = torch.mean( iwae_loss(log_weight, num_chains = S) )
        elbo_bound = torch.mean( log_weight )
    
    return iwae_bound, elbo_bound
