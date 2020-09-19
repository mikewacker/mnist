import gzip

import numpy as np

import idx

def load_mnist():
    """Loads the training and test sets for the MNIST data.

    Returns:
        X_train: training images as a (m_train, 28, 28) numpy array
        X_test: test images as a (m_test, 28, 28) numpy array
        y_train: training labels as a (m_train,) numpy array
        y_test: test labels as a (m_test,) numpy array
    """
    X_train = _load_idx(_X_TRAIN_PATH)
    X_test = _load_idx(_X_TEST_PATH)
    y_train = _load_idx(_Y_TRAIN_PATH)
    y_test = _load_idx(_Y_TEST_PATH)
    return X_train, X_test, y_train, y_test

####
# Data loading
####

_X_TRAIN_PATH = "data/train-images-idx3-ubyte.gz"
_X_TEST_PATH = "data/t10k-images-idx3-ubyte.gz"
_Y_TRAIN_PATH = "data/train-labels-idx1-ubyte.gz"
_Y_TEST_PATH = "data/t10k-labels-idx1-ubyte.gz"

def _load_idx(path):
    """Loads a single IDX file."""
    with gzip.open(path, "rb") as f:
        return idx.read_array(f)
