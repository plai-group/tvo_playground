# Adapted from
# https://github.com/tensorflow/models/tree/master/research/rebar and
# https://github.com/duvenaud/relax/blob/master/datasets.py

import logging
import numpy as np
import urllib.request
import pickle
from pathlib import Path

BINARIZED_MNIST_URL_PREFIX = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist'
BINARIZED_MNIST_DIR = Path('./binarized_mnist')
BINARIZED_MNIST_DIR.mkdir(exist_ok=True, parents=True)


def download_binarized_mnist():
    """Downloads the binarized MNIST dataset and saves to .npy files.

    Args:
        dir: directory where to save dataset
        url_prefix: prefix of url where to download from
        splits: list of url suffixes; subset of train, valid, test
    """
    binarized_mnist = []
    for split in ['train', 'valid', 'test']:
        filename = f'binarized_mnist_{split}.amat'
        url = f'{BINARIZED_MNIST_URL_PREFIX}/binarized_mnist_{split}.amat'
        path = BINARIZED_MNIST_DIR / filename
        urllib.request.urlretrieve(url, path)
        logging.info(f'Downloaded {url} to {path}')

        npy_filename = f'binarized_mnist_{split}.npy'
        npy_path = BINARIZED_MNIST_DIR / npy_filename
        with open(path, 'rb') as f:
            np.save(npy_path,
                    np.array([list(map(int, line.split()))
                              for line in f.readlines()], dtype='uint8'))
            logging.info(f'Saved to {npy_path}')

        binarized_mnist.append(np.load(npy_path))

    x_train, x_valid, x_test = binarized_mnist

    data = {'x_train': x_train,
            'x_valid': x_valid,
            'x_test': x_test}

    with open("binarized_mnist.pkl", 'wb') as file_handle:
        pickle.dump(data, file_handle)


download_binarized_mnist()
