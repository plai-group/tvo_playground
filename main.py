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

import src.ml_helpers as mlh
from src import assertions, util
from src.data_handler import get_data
from src.models import schedules
from src.models.model_handler import get_model

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

    The ProbModelBaseClass object is stateful and contains self.args,
    so hyperparameters are accessable to the model via self.args.hyper_param
    """
    # learning task
    model_name = 'continuous_vae'
    artifact_dir = './artifacts'
    data_dir = './data'

    # Model
    loss = 'elbo'
    hidden_dim = 200  # Hidden dimension of middle NN layers in vae
    latent_dim = 50  # Dimension of latent variable z
    integration = 'left'
    cuda = True
    num_stochastic_layers = 1
    num_deterministic_layers = 2
    learn_prior = False
    activation = None  # override Continuous VAE layers
    iw_resample = False  # whether to importance resample TVO proposals (WIP)

    # Hyper
    K = 5
    S = 10
    lr = 0.001
    log_beta_min = -1.09

    # Scheduling
    schedule = 'log'
    schedule_update_frequency = 1  # if 0, initalize once and never update
    per_sample = False # Update schedule for each sample
    per_batch = False # schedule update per batch

    # Recording
    record = False
    verbose = False
    dataset = 'mnist'

    # Training
    seed            = 1
    epochs          = 1000
    batch_size      = 1000
    valid_S         = 100
    test_S          = 5000
    test_batch_size = 1

    optimizer = "adam"
    checkpoint_frequency = int(epochs / 5)
    checkpoint = False
    checkpoint = checkpoint if checkpoint_frequency > 0 else False

    test_frequency = 20
    test_during_training = True
    test_during_training = test_during_training if test_frequency > 0 else False
    train_only = False
    save_grads = False

    phi_tag = 'encoder'
    theta_tag = 'decoder'


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
    args.partition_scheduler = schedules.get_partition_scheduler(args)
    args.partition = util.get_partition(args)

    # init data
    train_data_loader, test_data_loader = get_data(args)
    args.train_data_loader = train_data_loader
    args.test_data_loader = test_data_loader

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
