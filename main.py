import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import wandb
from sacred import Experiment

import src.handlers.ml_helpers as mlh
import assertions
from src.utils import schedules
#from src.models import schedules


#from src import assertions, util
from src.handlers.data_handler import get_data
from src.handlers.model_handler import get_model

# Use sacred for command line interface + hyperparams
# Use wandb for experiment tracking
ex = Experiment()
WANDB_PROJECT_NAME = 'SET_PROJECT_NAME'
if '--unobserved' in sys.argv:
    os.environ['WANDB_MODE'] = 'dryrun'


@ex.config
def my_config():
    """
    This specifies all the parameters for the experiment.
    Only native python objects can appear here (lists, string, dicts, are okay,
    numpy arrays and tensors are not). Everything defined here becomes
    a hyperparameter in the args object, as well as a column in omniboard.

    More complex objects are defined and manuipulated in the init() function
    and attached to the args object.

    The VAE or ProbModelBaseClass objects are stateful and contain self.args,
    so hyperparameters are accessible to the model via self.args.hyper_param
    """
    # learning task
    model_name = 'continuous_vae'
    artifact_dir = './artifacts'
    data_dir = './data'


    dataset = 'mnist'
    cuda = True
    loss = 'elbo'
    

    ''' Importance Sampling Params
        -------------------------- '''
    K = 5 # number of betas # replace with T = 5?
    S = 10 # number of AIS chains or importance samples # replace with N = 5?
    K_per_chain = 1 # multi-sample TVO or AIS with IWAE K-sample energy

    valid_S         = 100
    test_S          = 5000

    ''' Optimization Params
        -------------------------- '''
    optimizer = "adam"
    lr  =   0.001
    epochs          = 1000
    batch_size      = 1000
    test_batch_size = 1
    seed            = 1

    # for selective optimization of named parameters (i.e. [ param if phi_tag in param.name() for param in model.parameters() ] )
    phi_tag = 'encoder'
    theta_tag = 'decoder'


    ''' Architecture (Proposed) 
    ------------
    config:  'config.py'  # config file for architecture'''

    include_old_args = True
    # various (mostly architecture) args to be replaced
    if incl_old_args:

        hidden_dim = 200  # (prev: # Hidden dimension of middle NN layers in vae) 
        latent_dim = 50  # Dimension of latent variable z
        activation = None  # override Continuous VAE layers

        num_stochastic_layers = 1
        num_deterministic_layers = 2

        # should incorporate normalizing flow or Vamp prior
        learn_prior = False


        ''' Needs further infrastructure since UB (reverse) should sample from p(x,z)'''
        integration = 'left' # direction = 'fwd' / 'rev'.   


        iw_resample = False # never worked.  may consider operators which resample z_t according to chain weights

    
    ''' Flow Specification
    -------------------
        TO DO : absorb into architecture config
            --- would allow each flow to have its own arguments
    '''

    flows_per_beta = 2 # several flow transforms per π_β? 
    hidden_per_made_flow = 10*latent_dim
    linear_flow = None #'permute' # 'lu', etc.  (dims need to be consistent to eval Π q(z_j|x))
    spline_bins = 10



    ''' TVO Scheduling
        -------------- '''
    schedule = 'log'
    log_beta_min = -1.7 # defaults for log_uniform schedule
    beta_min = 0.02

    schedule_update_frequency = 1  # update every n epochs, (if 0, never update)
    
    if incl_old_args:
        # TO DO: revisit rescheduling schemes
        per_sample = False # Update schedule for each sample
        per_batch = False # schedule update per batch



    ''' AIS Parameters
        -------------- '''
    q = 1 # geometric path





    ''' Checkpointing / Testing / Recording
        ----------------------------------- '''
    record = False
    verbose = False

    checkpoint_frequency = int(epochs / 5)
    checkpoint = False
    checkpoint = checkpoint if checkpoint_frequency > 0 else False

    test_frequency = 20
    test_during_training = True
    test_during_training = test_during_training if test_frequency > 0 else False
    train_only = False
    save_grads = False




    if model_name == 'discrete_vae':
        dataset = 'binarized_mnist'
        # To match paper (see app. I)
        num_stochastic_layers = 3
        num_deterministic_layers = 0


    if model_name == 'bnn':
        dataset = 'fashion_mnist'

        bnn_mini_batch_elbo = True

        batch_size = 100 # To match tutorial (see: https://www.nitarshan.com/bayes-by-backprop/)
        test_batch_size = 5

        # This can still be overwritten via the command line
        S = 10
        test_S = 10
        valid_S = 10

    if model_name == 'pcfg':
        dataset = 'astronomers'
        ## to match rrws code
        batch_size = 2
        schedule = 'log'
        S = 20
        train_only = True # testing happens in training loop
        cuda = False
        epochs = 2000

        phi_tag = 'inference_network'
        theta_tag = 'generative_model'



def init(config, run):
    # general init
    args = SimpleNamespace(**config)
    args = assertions.validate_args(args)
    mlh.seed_all(args.seed)
    args._run = run
    args.wandb = wandb

    # init scheduler
    args.partition_scheduler = schedules.get_scheduling_fn(args)
    args.partition = schedules.get_initial_partition(args)

    # init data
    train_data_loader, test_data_loader = get_data(args)
    args.train_data_loader = train_data_loader
    args.test_data_loader = test_data_loader

    # ******** double check ***********
    args.input_dim = train_data_loader.dataset.image.shape[1]

    # init model
    model = get_model(train_data_loader, args)

    # init optimizer
    model.init_optimizer()

    return model, args


def log_scalar(**kwargs):
    assert "step" in kwargs, 'Step must be included in kwargs'
    step = kwargs.pop('step')
    wandb.log(kwargs)
    loss_string = " ".join(("{}: {:.4f}".format(*i) for i in kwargs.items()))
    print(f"Epoch: {step} - {loss_string}")


def train(model, args):
    for epoch in range(args.epochs):
        if mlh.is_schedule_update_time(epoch, args):
            args.partition = args.partition_scheduler(model, args)

        train_logpx, train_elbo = model.step_epoch(args.train_data_loader, step=epoch)

        log_scalar(train_elbo=train_elbo, train_logpx=train_logpx, step=epoch)

        if mlh.is_gradient_time(epoch, args):
            # Save grads
            grad_variance = util.calculate_grad_variance(model, args)
            log_scalar(grad_variance=grad_variance, step=epoch)

        if mlh.is_test_time(epoch, args):
            test_logpx, test_kl = model.test(args.test_data_loader, step=epoch)
            log_scalar(test_logpx=test_logpx, test_kl=test_kl, step=epoch)

        # ------ end of training loop ---------

    if args.train_only:
        test_logpx, test_kl = 0, 0

    results = {
        "test_logpx": test_logpx,
        "test_kl": test_kl,
        "train_logpx": train_logpx,
        "train_elbo": train_elbo
    }

    return results, model


@ex.automain
def command_line_entry(_run,_config):
    wandb_run = wandb.init(project = WANDB_PROJECT_NAME,
                            config = _config,
                              tags = [_run.experiment_info['name']])
    model, args = init(_config, _run)
    train(model, args)
