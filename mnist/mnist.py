import functools
import gzip
import math
import os

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from . import idx

_DIGITS = np.arange(10)

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

def score_predictions(y_true, y_pred):
    """Scores the predictions with a single metric and more detailed metrics.

    The single metric is accuracy as the macro-average of recall.

    Args:
        y_true: true labels as a (m,) numpy array
        y_pred: predicted labels as a (m,) numpy array

    Returns:
        score: accuracy as a macro-average of recall
        acc: accuracy/recall for each digit as a (10,) numpy array
        cm: row-normalized confusion matrix as a (10, 10) numpy array
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = np.sum(cm, axis=1, keepdims=True)
    cm = cm / cm_norm
    acc = cm[_DIGITS, _DIGITS]
    score = np.mean(acc)
    return score, acc, cm

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

def show_performance(y_true, y_pred):
    """Shows the overall performance of a model.

    Shows the overall score and the per-digit accuracy,
    both on a 0-100% scale and a 90-100% scale.
    Also plots two treemaps of the errors,
    starting from both the actual digit and the predicted digit.

    Args:
        y_true: true labels as a (m,) numpy array
        y_pred: predicted labels as a (m,) numpy array
    """
    # Process data.
    score, acc, cm = score_predictions(y_true, y_pred)
    title = "Score: {:.2%}".format(score)

    # Show data.
    axs = _create_performance_subplots()
    _plot_accuracy(axs[0], title, 0, score, acc)
    _plot_accuracy(axs[1], "90-100% Range", 0.9, score, acc)
    _plot_error_treemap(axs[2], "Errors by Actual Digit", cm)
    _plot_error_treemap(axs[3], "Errors by Predicted Digit", cm.T)

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
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, path)
    with gzip.open(filename, "rb") as f:
        return idx.read_array(f)

####
# Data processing
####

def _filter_images(X, y, digits):
    """Filters the images based on the digit."""
    if not _has_digits_filter(digits):
        return X, y
    mask = np.isin(y, digits)
    return X[mask], y[mask]

def _filter_predictions(X, y_true, y_pred, Y_prob, true_digits, pred_digits):
    """Filters the predictions based on the true and predicted digits."""
    masks = [
        np.isin(y, digits)
        for y, digits in [(y_true, true_digits), (y_pred, pred_digits)]
        if _has_digits_filter(digits)]
    if not masks:
        return X, y_true, y_pred, Y_prob
    mask = functools.reduce(lambda m1, m2: m1 & m2, masks)
    return X[mask], y_true[mask], y_pred[mask], Y_prob[mask]

def _has_digits_filter(digits):
    """Determines if the digits filter is enabled."""
    if digits is None:
        return False
    return np.isscalar(digits) or len(digits)

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
# Visualizing performance
####

def _create_performance_subplots():
    """Creates the subplots to visualize the performance."""
    figsize = (6, 6)
    gridspec_kw = {"height_ratios": (2.5, 2.5, 0.5, 0.5)}
    _, axs = plt.subplots(
        4, 1, constrained_layout=True,
        figsize=figsize, gridspec_kw=gridspec_kw)
    return axs

def _plot_accuracy(ax, title, min_acc, score, acc):
    """Plots the overall and per-digit accuracy."""
    # Set up the axes.
    ax.set_title(title)
    ax.axis((min_acc, 1, -0.5, 9.5))
    ax.invert_yaxis()
    if min_acc > 0:
        ax.spines["left"].set_visible(False)
    else:
        ax.spines["left"].set_linewidth(0.5)
    ax.spines["right"].set_visible(0.5)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)

    xticks = np.linspace(min_acc, 1, 11)
    xticklabels = [format(x, ".0%") for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(_DIGITS)
    ax.tick_params(left=False, top=True, width=0.5)

    # Plot the accuracy.
    color = [_digit_color(digit) for digit in _DIGITS]
    ax.barh(_DIGITS, acc, color=color, height=1)
    if score >= min_acc:
        ax.axvline(score, color=_OTHER_COLOR, lw=0.5)

    # Add the labels.
    for digit in _DIGITS:
        x = 0.11 + 0.89 * min_acc
        y = digit + 0.05
        value = acc[digit]
        label = format(value, ".2%")
        ax.text(x, y, label, ha="right", va="center")

def _plot_error_treemap(ax, title, cm):
    """Plots a treemap for the errors."""
    # Set up the axes.
    ax.set_title(title)
    ax.axis((0, 1, 0, 2))
    ax.invert_yaxis()
    ax.axis(False)

    # Plot the treemap.
    error = np.sum(cm) - np.sum(cm[_DIGITS, _DIGITS])
    major_offset = 0
    minor_offset = 0
    for digit1 in _DIGITS:
        for digit2 in _DIGITS:
            if digit1 == digit2:
                continue
            minor_size = cm[digit1, digit2] / error
            _plot_error_part(ax, 1, digit2, minor_offset, minor_size)
            minor_offset += minor_size
        major_size = minor_offset - major_offset
        _plot_error_part(ax, 0, digit1, major_offset, major_size)
        major_offset = minor_offset

def _plot_error_part(ax, row, digit, offset, size):
    """Plots a single part of the error."""
    # Plot the error part.
    x = [offset, offset + size]
    y1 = [row, row]
    y2 = [row + 1, row + 1]
    color = _digit_color(digit)
    ax.fill_between(x, y1, y2, color=color, lw=0)

    # Label the part if it is large enough.
    if size < 0.03:
        return
    x = offset + size / 2
    y = row + 0.55
    label = format(digit, "d")
    ax.text(x, y, label, ha="center", va="center")

####
# Visualizing predictions
####

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
    ax.axis((-0.5, 9.5, 0, 1))
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["top"].set_linewidth(0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plots the bars.
    color = 10 * [_OTHER_COLOR]
    color[int(label)] = _pred_color(True)
    color[int(pred)] = _pred_color(pred == label)
    ax.bar(_DIGITS, prob, color=color, width=1)

    # Add the labels.
    for digit in _DIGITS:
        x = digit - 0.07
        label_text = format(digit, "d")
        color = "black" if digit == label else _OTHER_COLOR
        ax.text(x, 0.8, label_text, color=color, ha="center", va="center")

####
# Color scheme
####

_CMAP = plt.get_cmap("coolwarm")

_OTHER_COLOR = "silver"

def _pred_color(correct):
    """Gets the color for a correct or incorrect prediction."""
    return _CMAP(0.1) if correct else _CMAP(0.9)

def _digit_color(digit):
    """Gets the color for the digit."""
    value = (digit + 0.5) / 10
    return _CMAP(value)
