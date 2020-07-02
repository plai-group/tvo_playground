# Assertions

SCHEDULES         = ['log', 'linear', 'moments']
PARTITONS         = ['left','right','trapz']

TVO_LOSSES        = ['tvo','tvo_reparam']
DUAL_LOSSES       = ['wake-wake', 'wake-sleep', 'tvo_reparam', 'iwae_dreg']
REQUIRES_REPARAM  = ['elbo', 'iwae', 'tvo_reparam', 'iwae_dreg']
ALL_LOSSES        = REQUIRES_REPARAM + ['reinforce','tvo', 'vimco','wake-wake','wake-sleep']

DISCRETE_MODELS   = ['discrete_vae','pcfg']
CONTINUOUS_MODELS = ['continuous_vae', 'bnn']
ALL_MODELS        = DISCRETE_MODELS + CONTINUOUS_MODELS

BINARIZED_DATASETS = ['binarized_mnist','binarized_omniglot']
ALL_DATASETS       = BINARIZED_DATASETS + ['fashion_mnist','mnist','kuzushiji_mnist','omniglot']
PCFGS             = ['astronomers', 'brooks', 'minienglish', 'polynomial', 'quadratic', 'sids']

def validate_hypers(args):
    assert args.schedule in SCHEDULES, f"schedule cannot be {args.schedule}"
    assert args.integration in PARTITONS, f"integration cannot be {args.integration}"
    assert args.loss in ALL_LOSSES, f"loss cannot be {args.loss} "
    assert args.model_name in ALL_MODELS, f" model cannot be {args.model}"
    assert args.dataset in ALL_DATASETS, f" dataset cannot be {args.dataset} "

    if args.model_name in DISCRETE_MODELS:
        assert args.loss not in REQUIRES_REPARAM, f"loss can't be {args.loss} with {args.model_name}"

    if args.schedule != 'log':
        assert args.loss in TVO_LOSSES, f"{args.loss} doesn't require a partition schedule scheme"

    if args.model_name == 'discrete_vae':
        assert args.dataset in BINARIZED_DATASETS, f" dataset cannot be {args.dataset} with {args.model_name}"

    if args.loss in DUAL_LOSSES:
        assert not args.save_grads, 'Grad variance not able to handle duel objective methods yet'


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
    else:
        raise ValueError("Unknown learning task")

    return data_path
