from pathlib import Path
import numpy as np
from src.ml_helpers import detect_cuda
# Assertions

SCHEDULES = ['log', 'linear', 'moments','gp_bandits']
PARTITONS = ['left','right','trapz']

TVO_LOSSES       = ['tvo','tvo_reparam']
DUAL_LOSSES      = ['wake-wake', 'wake-sleep', 'tvo_reparam', 'iwae_dreg']
DISCRETE_LOSSES  = ['tvo','tvo_reparam', 'wake-wake', 'wake-sleep', 'reinforce', 'vimco']
REQUIRES_REPARAM = ['elbo', 'iwae', 'tvo_reparam', 'iwae_dreg']
ALL_LOSSES       = REQUIRES_REPARAM + ['reinforce','tvo', 'vimco','wake-wake','wake-sleep']

DISCRETE_MODELS   = ['discrete_vae','pcfg']
CONTINUOUS_MODELS = ['continuous_vae', 'bnn']
ALL_MODELS        = DISCRETE_MODELS + CONTINUOUS_MODELS

BINARIZED_DATASETS = ['binarized_mnist','binarized_omniglot']
PCFGS              = ['astronomers', 'brooks', 'minienglish', 'polynomial', 'quadratic', 'sids']
ALL_DATASETS       = BINARIZED_DATASETS + ['fashion_mnist','mnist','kuzushiji_mnist','omniglot']

def validate_hypers(args):
    assert args.schedule in SCHEDULES, f"schedule cannot be {args.schedule}"
    assert args.integration in PARTITONS, f"integration cannot be {args.integration}"
    assert args.loss in ALL_LOSSES, f"loss cannot be {args.loss} "
    assert args.model_name in ALL_MODELS, f" model cannot be {args.model}"
    assert args.dataset in ALL_DATASETS + PCFGS, f" dataset cannot be {args.dataset} "

    if args.model_name in DISCRETE_MODELS:
        assert args.loss not in REQUIRES_REPARAM, f"loss can't be {args.loss} with {args.model_name}"

    if args.schedule != 'log':
        assert args.loss in TVO_LOSSES, f"{args.loss} doesn't require a partition schedule scheme"

    if args.model_name == 'discrete_vae':
        assert args.dataset in BINARIZED_DATASETS, f" dataset cannot be {args.dataset} with {args.model_name}"

    if args.loss in DUAL_LOSSES:
        assert not args.save_grads, 'Grad variance not able to handle duel objective methods yet'

    if args.model_name == 'pcfg':
        assert args.dataset in PCFGS, f"dataset must be one of {PCFGS}"
        assert args.loss in DISCRETE_LOSSES, f"loss can't be {args.loss} with {args.model_name}"



def validate_dataset_path(args):
    model_name = args.model_name
    dataset = args.dataset
    if model_name in ['discrete_vae', 'continuous_vae']:
        if dataset == 'fashion_mnist':
            data_path = args.data_dir + '/fashion_mnist.pkl'
        elif dataset == 'mnist':
            data_path = args.data_dir + '/mnist.pkl'
        elif dataset == 'omniglot':
            data_path = args.data_dir + '/omniglot.pkl'
        elif dataset == 'kuzushiji_mnist':
            data_path = args.data_dir + '/kuzushiji_mnist.pkl'
        elif dataset == 'binarized_mnist':
            data_path = args.data_dir + '/binarized_mnist.pkl'
        elif dataset == 'binarized_omniglot':
            data_path = args.data_dir + '/binarized_omniglot.pkl'
    elif model_name in ['bnn']:
        if dataset == 'fashion_mnist':
            data_path = args.data_dir + '/fmnist/'
    elif model_name in ['pcfg']:
        data_path = args.data_dir + f'/pcfgs/{dataset}_pcfg.json'
    else:
        raise ValueError("Unknown model name")

    return Path(data_path)

def validate_args(args):
    # check assertions
    validate_hypers(args)

    # check data path
    args.data_path = validate_dataset_path(args)

    # add bandit memory args necessary
    if args.schedule == 'gp_bandits':
        args.betas_all = np.empty((0, args.K+1), float)
        args.logtvopx_all = [] # for storage
        args.average_y = []
        args.X_ori = np.empty((0, args.K + 1), float)
        args.Y_ori = []

    # check artifact_dir
    Path(args.artifact_dir).mkdir(exist_ok=True)

    # check cuda
    args = detect_cuda(args)

    return args

