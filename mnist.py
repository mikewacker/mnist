import functools
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

def preprocess_flat(X):
    """Flattens the images and scales them from 0.0 to 1.0.

    Args:
        X: images as a (m, 28, 28) numpy array

    Returns:
        X: scaled images as a (m, 784) numpy array
    """
    X = X.reshape((-1, 784))
    return X / 255

def preprocess_channel(X):
    """Converts the images to 1 channel and scales them from 0.0 to 1.0.

    Args:
        X: images as a (m, 28, 28) numpy array

    Returns:
        X: scaled images as a (m, 28, 28, 1) numpy array
    """
    X = X.reshape((-1, 28, 28, 1))
    return X / 255

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
    Y_prob[np.arange(m), y_pred] = 1.0
    return Y_prob

def show_images(X, y, *, digits=None, size=64):
    """Shows images and their labels.

    Args:
        X: images as a (m, 28, 28) numpy array
        y: labels as a (m,) numpy array
        digits: digit or list of digits to show
        size: number of samples to show
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

def show_predictions(
    X, y_true, y_pred, Y_prob,
    *, true_digits=None, pred_digits=None, size=16):
    """Shows predictions, including the probability for each digit.

    The two columns are respectively for correct and incorrect predictions.

    Args:
        X: images as a (m, 28, 28) numpy array
        y_true: true labels as a (m,) numpy array
        y_pred: predicted labels as a (m,) numpy array
        Y_prob: predicted probabilities as a (m, 10) numpy array
        true_digits: true digit or list of true digits to show
        pred_digits: predicted digit or list of predicted digits to show
        size: number of correct and incorrect samples to show
    """
    # Process data.
    X, y_true, y_pred, Y_prob = _filter_predictions(
        X, y_true, y_pred, Y_prob, true_digits, pred_digits)
    splits = _split_predictions(X, y_true, y_pred, Y_prob)
    splits = [_sample_rows(*split, size=size) for split in splits]
    m_max = max(X.shape[0] for X, _, _, _ in splits)

    # Show data.
    axs = _create_predictions_subplots(m_max)
    for i, pred_tf_axs in enumerate(_generate_predictions_axs(axs)):
        for pred_axs, (X, y_true, y_pred, Y_prob) in zip(pred_tf_axs, splits):
            m = X.shape[0]
            if i >= m:
                [ax.remove() for ax in pred_axs]
                continue
            _plot_image(pred_axs[0], X[i])
            _plot_true_label(pred_axs[1], y_true[i])
            _plot_predicted_label(pred_axs[2], y_true[i], y_pred[i])
            _plot_probabilities(pred_axs[3], y_true[i], y_pred[i], Y_prob[i])

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

def _filter_predictions(X, y_true, y_pred, Y_prob, true_digits, pred_digits):
    """Filters the predictions based on the true and predicted digits."""
    masks = [
        np.isin(y, digits)
        for y, digits in [(y_true, true_digits), (y_pred, pred_digits)]
        if digits is not None and (np.isscalar(digits) or len(digits))]
    if not masks:
        return X, y_true, y_pred, Y_prob
    mask = functools.reduce(lambda m1, m2: m1 & m2, masks)
    return X[mask], y_true[mask], y_pred[mask], Y_prob[mask]

def _split_predictions(X, y_true, y_pred, Y_prob):
    """Splits the data based on whether the prediction was correct."""
    t_mask = y_true == y_pred
    f_mask = ~t_mask
    split_arrays = [(A[t_mask], A[f_mask]) for A in [X, y_true, y_pred, Y_prob]]
    return list(zip(*split_arrays))

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
    ax.text(0, -0.17, text, color=color, fontsize=32, ha="center", va="center")

####
# Visualizing predictions
####

_DIGITS = np.arange(10)

def _create_predictions_subplots(m):
    """Creates the subplots to visualize m correct and incorrect predictions."""
    col_widths = (0.5, 0.5, 0.5, 1.5, 0.25, 0.5, 0.5, 0.5, 1.5)
    axs = _create_packed_subplots(m, col_widths)
    return _remove_border_columns(axs, 4)

def _generate_predictions_axs(axs):
    """Generates the list of axes for each pair of predictions."""
    num_rows = axs.shape[0]
    for row in range(num_rows):
        yield axs[row, 0:4], axs[row, 4:8]

def _plot_predicted_label(ax, label, pred):
    """Plots an "image" of the predicted label."""
    color = _pred_color(label == pred)
    _plot_label(ax, pred, color)

def _plot_probabilities(ax, label, pred, prob):
    """Plots the predicted probability for each digit."""
    # Set up the axes.
    ax.set(
        xlim=(-0.5, 9.5),
        ylim=(0, 1),
        xticks=[],
        yticks=[])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["top"].set_linewidth(0.5)

    # Plots the bars.
    color = 10 * [_OTHER_COLOR]
    color[int(label)] = _pred_color(True)
    color[int(pred)] = _pred_color(pred == label)
    ax.bar(_DIGITS, prob, color=color, width=1)

    # Add the labels.
    for digit in _DIGITS:
        color = "black" if digit == label else _OTHER_COLOR
        label_text = format(digit, "d")
        ax.text(
            digit - 0.07, 0.8, label_text, color=color,
            ha="center", va="center")

####
# Color scheme
####

_CMAP = plt.get_cmap("coolwarm")

_OTHER_COLOR = "silver"

def _pred_color(correct):
    """Gets the color for a correct or incorrect prediction."""
    return _CMAP(0.1) if correct else _CMAP(0.9)
