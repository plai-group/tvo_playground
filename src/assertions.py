# Assertions
def validate_hypers(args):
    assert args.schedule in [
        'log',
        'linear',
        'bq',
        'auto_bq'], f"schedule cannot be {args.schedule}"

    assert args.integration in [
        'left',
        'right',
        'trap',
        'single'], f"integration cannot be {args.integration}"

    assert args.loss in [
        'reinforce',
        'elbo',
        'iwae',
        'tvo',
        'vimco',
        'wake-wake',
        'wake-sleep'], f"loss cannot be {args.loss} "

    assert args.learning_task in [
        'continuous_vae',
        'discrete_vae'], f" learning_task cannot be {args.learning_task}"

    assert args.dataset in [
        'fashion_mnist',
        'mnist',
        'kuzushiji_mnist',
        'omniglot',
        'binarized_mnist',
        'binarized_omniglot'], f" dataset cannot be {args.dataset} "

    if args.schedule != 'log':
        assert args.loss == 'tvo', f"{args.loss} doesn't require a partition schedule scheme"

    if args.learning_task in ['discrete_vae']:
        assert args.loss not in [
            'elbo', 'iwae'], f"loss can't be {args.loss} with {args.learning_task}"

    if args.learning_task == 'discrete_vae':
        assert args.dataset in ['binarized_mnist', 'binarized_omniglot'], \
            f" dataset cannot be {args.dataset} with {args.learning_task}"

    if args.loss in ['wake-wake', 'wake-sleep']:
        assert not args.save_grads, 'Grad variance not able to handle duel objective methods yet'

    # Add an assertion everytime you catch yourself making a silly hyperparameter mistake so it doesn't happen again


def validate_dataset_path(args):
    learning_task = args.learning_task
    dataset = args.dataset

    if learning_task in ['discrete_vae', 'continuous_vae']:
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
