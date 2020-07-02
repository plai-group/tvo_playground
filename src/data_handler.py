import pickle
import logging
import numpy as np
from pathlib import Path
import torch.utils.data
from src.ml_helpers import tensor, get_data_loader
from src.models.pcfg import GenerativeModel as PCFGGenerativeModel
from src.models.pcfg_util import read_pcfg
from torchvision import datasets, transforms
from torch.utils.data import Dataset, IterableDataset
from skimage import color, io as imageio, transform


class StochasticMNIST(Dataset):
    def __init__(self, image):
        super(StochasticMNIST).__init__()
        self.image = image

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, idx):
        return (torch.bernoulli(self.image[idx, :]), )


class PixelIntensity(Dataset):
    def __init__(self, image):
        super(PixelIntensity).__init__()
        self.image = image

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, idx):
        return (self.image[idx, :], )


class Synthetic(Dataset):
    def __init__(self, X, y=None):
        super(Synthetic).__init__()
        self.X = X
        self.y = y

        if X.ndim == 1:
            self.X = self.X.unsqueeze(1)

        if y is not None:
            assert self.X.shape[0] == self.y.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is not None:
            return (self.X[idx, :], self.y[idx])
        else:
            return (self.X[idx, :], )


def make_continuous_vae_data(args):
    # read data
    with open(args.data_path, 'rb') as file_handle:
        data = pickle.load(file_handle)

    train_image = data['train_image']
    test_image = data['test_image']

    # See page 6, footnote 2 here: https://arxiv.org/pdf/1509.00519.pdf
    train_image = StochasticMNIST(tensor(train_image, args))
    test_image = StochasticMNIST(tensor(test_image, args))

    train_data_loader = get_data_loader(train_image, args.batch_size, args)
    test_data_loader = get_data_loader(test_image, args.test_batch_size, args)
    return train_data_loader, test_data_loader


def make_discrete_vae_data(args):
    """
    Annoyingly the continuous and discrete vae literature uses different train/val/test/split.
    For continuous we use 60k train / 10k test
    in accordance with IWAE paper: https://arxiv.org/pdf/1509.00519.pdf
    For discrete we use 50k train / 10k validation / 10k test
    in accordance with VIMCO paper: https://arxiv.org/pdf/1602.06725.pdf
    We don't use the 10k validation to be consistent w/ continuous case
    """
    with open(args.data_path, 'rb') as file_handle:
        data = pickle.load(file_handle)

    train_image = PixelIntensity(tensor(data['x_train'], args))
    test_image = PixelIntensity(tensor(data['x_test'], args))

    train_data_loader = get_data_loader(train_image, args.batch_size, args)
    test_data_loader  = get_data_loader(test_image, args.test_batch_size, args)

    return train_data_loader, test_data_loader


def make_bnn_data(args):
    LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_data_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            args.data_path, train=True, download=True,
            transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **LOADER_KWARGS)
    test_data_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            args.data_path, train=False, download=True,
            transform=transforms.ToTensor()),
        batch_size=args.test_batch_size, shuffle=False, **LOADER_KWARGS)

    return train_data_loader, test_data_loader


class PCFGDataset(Dataset):
    def __init__(self, path, N=1000):
        super(PCFGDataset).__init__()
        grammar, true_production_probs = read_pcfg(path)
        self.true_generative_model = PCFGGenerativeModel(grammar, true_production_probs)
        # data comes from true_generative_model so we have an infinite amount of it.
        # Set len = 1000 arbitrarily
        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.true_generative_model.sample_obs()

def make_pcfg_data(args):
    # instantiate a gen model w/ true production probabilities to create data
    dataset = PCFGDataset(args.data_path, N=args.batch_size)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        batch_sampler=None,
        shuffle=False,
        collate_fn=lambda x: (x,) # this is important otherwise default collate_fn messes up batching
        )

    return loader, loader


def get_data(args):
    if args.model_name == 'continuous_vae':
        return make_continuous_vae_data(args)
    elif args.model_name == 'discrete_vae':
        return make_discrete_vae_data(args)
    elif args.model_name == 'bnn':
        return make_bnn_data(args)
    elif args.model_name == 'pcfg':
        return make_pcfg_data(args)
    else:
        raise ValueError("{} is an invalid learning task".format(args.model_name))
