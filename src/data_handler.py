# Adapted from
# https://github.com/tensorflow/models/tree/master/research/rebar and
# https://github.com/duvenaud/relax/blob/master/datasets.py

import pickle
import logging
import numpy as np
from pathlib import Path
import torch.utils.data
from src.ml_helpers import tensor, get_data_loader
from torch.utils.data import Dataset
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
    # read data
    with open(args.data_path, 'rb') as file_handle:
        data = pickle.load(file_handle)

    train_image = PixelIntensity(tensor(data['x_train'], args))
    test_image = PixelIntensity(tensor(data['x_test'], args))

    train_data_loader = get_data_loader(train_image, args.batch_size, args)
    test_data_loader  = get_data_loader(test_image, args.test_batch_size, args)

    return train_data_loader, test_data_loader

def get_data(args):
    if args.model_name == 'continuous_vae':
        return make_continuous_vae_data(args)
    elif args.model_name == 'discrete_vae':
        return make_discrete_vae_data(args)
    else:
        raise ValueError(
            "{} is an invalid learning task".format(args.model_name))
