import gzip
import os

import numpy as np

from . import idx

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

def preprocess_flat(X):
    """Flattens the images and scales them from 0.0 to 1.0.

    Args:
        X: images as a (m, 28, 28) numpy array

    Returns:
        scaled images as a (m, 784) numpy array
    """
    return _preprocess(X, (784,))

def preprocess_channel(X):
    """Converts the images to 1 channel and scales them from 0.0 to 1.0.

    Args:
        X: images as a (m, 28, 28) numpy array

    Returns:
        scaled images as a (m, 28, 28, 1) numpy array
    """
    return _preprocess(X, (28, 28, 1))

def to_pred(Y_prob):
    """Converts probabilistic predictions to discrete predictions.

    Args:
        Y_prob: predicted probabilities as a (m, 10) numpy array

    Returns:
        predicted labels as a (m,) numpy array
    """
    return np.argmax(Y_prob, axis=1)

def to_onehot_prob(y_pred):
    """Converts discrete predictions to (one-hot) probabilistic predictions.

    Args:
        y_pred: predicted labels as a (m,) numpy array

    Returns:
        predicted probabilities as a (m, 10) numpy array
    """
    m = y_pred.shape[0]
    Y_prob = np.zeros((m, 10))
    Y_prob[np.arange(m), y_pred] = 1.
    return Y_prob

def score_predictions(y_true, y_pred):
    """Scores the predictions with a single metric and more detailed metrics.

    The single metric is accuracy.

    Args:
        y_true: true labels as a (m,) numpy array
        y_pred: predicted labels as a (m,) numpy array

    Returns:
        score: overall accuracy
        acc: accuracy/recall for each digit as a (10,) numpy array
        cm: confusion matrix as a (10, 10) numpy array
    """
    cm = _confusion_matrix(y_true, y_pred)
    acc = cm[_DIGITS, _DIGITS] / np.sum(cm, axis=1)
    score = np.mean(y_true == y_pred)
    return score, acc, cm

####
# Implementation
####

_DIGITS = np.arange(10)

_X_TRAIN_PATH = "train-images-idx3-ubyte.gz"
_X_TEST_PATH = "t10k-images-idx3-ubyte.gz"
_Y_TRAIN_PATH = "train-labels-idx1-ubyte.gz"
_Y_TEST_PATH = "t10k-labels-idx1-ubyte.gz"

def _load_idx(filename):
    """Loads a single IDX file."""
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "data", filename)
    with gzip.open(filename, "rb") as f:
        return idx.read_array(f)

def _preprocess(X, shape):
    """Reshapes and scales inputs."""
    m = X.shape[0]
    X = X.reshape((m, *shape))
    return X / 255

def _confusion_matrix(y_true, y_pred):
    """Creates a confusion matrix."""
    cm = np.zeros((10, 10))
    for true_digit in range(10):
        y_pred_digit = y_pred[y_true == true_digit]
        digits, counts = np.unique(y_pred_digit, return_counts=True)
        for digit, count in zip(digits, counts):
            cm[true_digit, digit] = count
    return cm
