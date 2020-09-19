import gzip
import math

import numpy as np
import matplotlib.pyplot as plt

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

def show_images(X, y, *, digits=None, size=64):
    """Shows images and their labels.

    Args:
        X: images as a (m, 28, 28) numpy array
        y: labels as a (m,) numpy array
        digits: (optional) digit or list of digits to show
        size: (optional) number of samples to show, defaults to 64
    """
    # Process data.
    X, y = _filter_images(X, y, digits)
    X, y = _sample_rows(X, y, size=size)
    m = X.shape[0]

    # Show data.
    axs = _create_images_subplots(m)
    for i, image_axs in enumerate(_generate_images_axs(axs)):
        if i >= m:
            [ax.remove() for ax in image_axs]
            continue
        _plot_image(image_axs[0], X[i])
        _plot_true_label(image_axs[1], y[i])

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

####
# Data processing
####

def _filter_images(X, y, digits):
    """Filters the images based on the digit."""
    if digits is None:
        return X, y
    mask = np.isin(y, digits)
    return X[mask], y[mask]

def _sample_rows(*arrays, size):
    """Samples rows, taking the same rows for each array."""
    m = arrays[0].shape[0]
    if size > m:
        return arrays
    indices = np.random.choice(np.arange(m), size=size, replace=False)
    return tuple(array[indices] for array in arrays)

####
# Visualizing images
####

def _create_images_subplots(m):
    """Creates the subplots to visualize m images and their labels."""
    num_rows = math.ceil(m / 4)
    col_widths = (0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5)
    axs = _create_packed_subplots(num_rows, col_widths)
    return _remove_border_columns(axs, 2)

def _create_packed_subplots(num_rows, col_widths):
    """Creates packed subplots."""
    width = sum(col_widths) + 0.25
    height = num_rows / 2 + 0.25
    lr_pad = 0.125 / width
    bt_pad = 0.125 / height
    _, axs = plt.subplots(
        num_rows, len(col_widths), squeeze=False,
        figsize=(width, height), gridspec_kw={"width_ratios": col_widths})
    plt.subplots_adjust(
        left=lr_pad, bottom=bt_pad, right=1-lr_pad, top=1-bt_pad,
        wspace=0, hspace=0)
    return axs

def _remove_border_columns(axs, col_size):
    """Removes the border columns, returning the remaining columns."""
    num_cols = axs.shape[1]
    border_indices = list(range(col_size, num_cols, col_size+1))
    border_axs = axs[:, border_indices].reshape(-1)
    [ax.remove() for ax in border_axs]
    data_indices = [col for col in range(num_cols) if col not in border_indices]
    return axs[:, data_indices]

def _generate_images_axs(axs):
    """Generates the list of axes for each image."""
    num_rows = axs.shape[0]
    num_cols = axs.shape[1] // 2
    for row in range(num_rows):
        for col in range(num_cols):
            col_offset = 2 * col
            col_end = col_offset + 2
            yield axs[row, col_offset:col_end]

def _plot_image(ax, image):
    """Plots an image of a handwritten digit."""
    ax.axis(False)
    ax.imshow(image, cmap="gray_r", vmin=0, vmax=255)

def _plot_true_label(ax, label):
    """Plots an "image" of the true label."""
    _plot_label(ax, label, "black")

def _plot_label(ax, label, color):
    """Plots an "image" of the label."""
    ax.axis((-1, 1, -1, 1))
    ax.axis(False)
    text = format(label, "d")
    ax.text(0, -0.15, text, color=color, fontsize=32, ha="center", va="center")
