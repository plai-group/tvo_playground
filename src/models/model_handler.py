from src.models.vaes import DiscreteVAE, ContinuousVAE
from src.models.bnn import BayesianNetwork
from src.models.pcfg import PCFG
from src.pcfg_util import read_pcfg

def get_model(train_data_loader, args):
    if args.learning_task == 'continuous_vae':
        D = train_data_loader.dataset.image.shape[1]
        model = ContinuousVAE(D, args)
    elif args.learning_task == 'discrete_vae':
        D = train_data_loader.dataset.image.shape[1]
        train_obs_mean = train_data_loader.dataset.image.mean(0)
        model = DiscreteVAE(D, args, train_obs_mean)
    elif args.learning_task == 'bnn':
        w, h = train_data_loader.dataset.train_data.shape[1:]
        D = w * h
        num_batches = len(args.train_data_loader)
        model = BayesianNetwork(D, num_batches, args)
    elif args.learning_task == 'pcfg':
        grammar, true_production_probs = read_pcfg(args.data_path)
        model = PCFG(grammar, args)
        # PCFG not set up for gpu
        return model
    else:
        raise ValueError("Incorrect learning task: {} not valid".format(args.learning_task))
    if args.device.type == 'cuda':
        model.cuda()

    return model
