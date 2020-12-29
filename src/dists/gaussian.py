
import torch.nn as nn
import numpy as np
import torch
from math import pi as pi
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.autograd import Variable

import src.utils.math_utils as math_utils

def log_normal_likelihood(x, mean, logvar, sum_dim = None):
    """Implementation WITH constant
    based on https://github.com/lxuechen/BDMC/blob/master/utils.py

    Args:
        x: [B,Z]
        mean,logvar: [B,Z]

    Returns:
        output: [B]
    """

    dim = list(mean.size())[-1] if torch.is_tensor(mean) else 1
    sumdim = 1 if dim > 1 else 0
    sum_dim = sum_dim if sum_dim is not None else sumdim

    #logvar = torch.zeros(mean.size()) + logvar
    return -0.5 * ((logvar + (x - mean)**2 / torch.exp(logvar)).sum(sum_dim) +
                   torch.log(torch.tensor(2 * pi)) * dim)


class StandardNormal():
    def __init__(self, args):
        #super().__init__()
        self.args = args
        self.shape = ( args.batch_size, args.S, args.latent_dim ) \
                    if args.K_per_chain == 1 else \
                    ( args.K_per_chain, args.batch_size, args.S, args.latent_dim ) 

    def log_prob(self, z):
        return log_normal_likelihood(z, torch.zeros_like(z), torch.zeros_like(z), -1)
        #Normal(torch.zeros_like(z), torch.zeros_like(z)).log_prob(z)

    def sample(self, S):
        return Normal(torch.zeros_like(self.shape), torch.ones_like(self.shape)).sample(S).transpose(1,0)


class ConditionalGaussian(nn.Module):
    """Conditional Gaussian distribution, where the mean and variance are
    parametrized with neural networks."""

    ''' Initially intended for Encoders, but might also be used for Gaussian observation model '''
    def __init__(self, encoder_fn, args):
        super(ConditionalGaussian, self).__init__()
        self.encoder = encoder_fn
        self.args = args
        self.stop_grad = 'reparam' in self.args.loss or 'dreg' in self.args.loss or 'stl' in self.args.loss

    def safe_encode(self, latents, size = None):
        if isinstance(latents, tuple):
            return latents[0], latents[1]
        else:
            size = size if size is not None else self.args.latent_dim
            if latents.shape[-1] == self.args.input_dim * 2:
                size = self.args.input_dim

            return latents[:,:size], latents[:, size:]

    def forward(self, x):
        self.mu, self.logvar = self.safe_encode(self.encoder(x))
        return self
        #return self.mu, self.logvar

    def conditional_sample(self, x, S=1):
        self.mu, self.logvar = self.safe_encode(self.encoder(x))
        return self.sample(self.mu, self.logvar, S)

    def mean(self, params= None):
        return self.mu 
        #m, lv = self.net(x)
        #return m

    def sample(self, S=1, mu=None, logvar = None, rsample=False):
        mu = self.mu if mu is None else mu
        logvar = self.logvar if logvar is None else logvar


        if rsample:
            z = torch.distributions.Normal( mu, torch.exp(.5*logvar) ).rsample([S,]).transpose(1,0)
        else:
            z = torch.distributions.Normal( mu, torch.exp(.5*logvar) ).sample([S,]).transpose(1,0)

        z.requires_grad = True
        return z
        #return z.permute(1,2,0)
        # and then flatten later?

        #std = torch.exp(0.5 * logvar)
        #eps = torch.randn_like(std)
        #return mu + std * eps

    def sample_and_log_prob(self, x, S=1):
        z = self.conditional_sample(x, S)

        log_prob = self.log_prob(z, self.mu, self.logvar)
        return z.reshape((-1, self.mu.shape[1])), log_prob.reshape(-1,)# self.mu.shape[1])
    
    def sample_and_log_prob_mu(self, mu, logvar, S=1):
        z = self.sample(mu, logvar, S)
        log_prob = self.log_prob(z, mu, logvar)
        return z, log_prob

    def log_prob(self, z, mu = None, logvar = None, stop_grad = None): 
        stop_grad = stop_grad if stop_grad is not None else self.stop_grad

        mu = self.mu if mu is None else mu
        logvar = self.logvar if logvar is None else logvar

        while len(mu.shape) < len(z.shape):
            mu = mu.unsqueeze(1)
            logvar = logvar.unsqueeze(1)

        if stop_grad:
            mu = mu.detach()
            logvar = logvar.detach() 

        lnl = log_normal_likelihood(z, mu, logvar, -1)
        return lnl

    def prior_kl(self, mu, logvar):
        """ Computes KL(q(z|x) || p(z)) assuming p(z) is N(0, I). """
        kl = -0.5 * math_utils.sum_except_batch(1 + logvar - (mu ** 2) - torch.exp(logvar), num_batch_dims=1)
        return torch.mean(kl, dim=0)
