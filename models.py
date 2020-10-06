import time

import numpy as np
import matplotlib.pyplot as plt

from mnist import mnist_data

def train_sklearn(model, X, y):
    """Trains a sklearn model."""
    _time_fn("Train", lambda: model.fit(X, y))

def evaluate_sklearn(model, X, y, prob=True):
    """Evaluates a sklearn model."""
    if prob:
        Y_prob = _time_fn("Predict", lambda: model.predict_proba(X))
        y_pred = mnist_data.to_pred(Y_prob)
    else:
        y_pred = _time_fn("Predict", lambda: model.predict(X))
        Y_prob = mnist_data.to_onehot_prob(y_pred)
    score, _, _ = mnist_data.score_predictions(y, y_pred)
    print("  Score: {:6.2%}".format(score))
    return y_pred, Y_prob

def train_nn(
    init_nn, X_train, X_test, y_train, y_test, *,
    start_epoch=0, seed=0, epoch_fn=None):
    """Trains the neural network for 20 epochs, returning the trained network.

    Starts the mini-batch size at 32, and then doubles it every 4 epochs.
    Saves the best epoch from the final 4 epochs as the official results.
    Can invoke a function after each epoch with the network and training time.
    """
    init_seed, epoch_seeds = _init_seeds(seed, start_epoch)
    nn = _init_nn(init_nn, start_epoch, init_seed)
    for epoch, epoch_seed in zip(range(start_epoch, _NUM_EPOCHS), epoch_seeds):
        _train_nn_epoch(nn, X_train, y_train, epoch, epoch_seed, epoch_fn)
    _save_best_epoch(nn, X_test, y_test)
    load_nn(nn)
    return nn

def evaluate_nn(nn, X, y):
    """Evaluates the neural network."""
    print(" Params: {:d}".format(nn.num_params))
    y_pred, Y_prob = _time_fn("Predict", lambda: nn.predict(X))
    score, _, _ = mnist_data.score_predictions(y, y_pred)
    cost = nn.cost(y)
    print("  Score: {:6.2%}".format(score))
    print("   Cost: {:.8f}".format(cost))
    return y_pred, Y_prob

def load_nn(nn, epoch=None):
    """Loads pre-trained weights."""
    filename = _nn_filename(nn.name, epoch)
    nn.load(filename)

class EpochTracker(object):
    """Tracks the progress after each epoch."""

    def __init__(self, X_train, X_test, y_train, y_test):
        """Initializes the tracker with the data."""
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

        self._total_duration = 0.
        self._train_scores = []
        self._train_costs = []
        self._test_scores = []
        self._test_costs = []

    def evaluate_epoch(self, nn, duration):
        """Callback to evaluate the network after each epoch."""
        print(".", end="")
        self._total_duration += duration
        self._evaluate(
            nn, self._X_train, self._y_train,
            self._train_scores, self._train_costs)
        self._evaluate(
            nn, self._X_test, self._y_test,
            self._test_scores, self._test_costs)

    def show_training(self):
        """Shows the training process."""
        print()
        self._print_training_time()
        _, (ax_score, ax_cost) = plt.subplots(
            2, 1, figsize=(6, 8), constrained_layout=True)
        self._plot_metric(
            ax_score, "Accuracy", self._train_scores, self._test_scores)
        self._plot_metric(ax_cost, "Cost", self._train_costs, self._test_costs)

    def _evaluate(self, nn, X, y, scores, costs):
        """Evaluates the neural network against the data."""
        y_pred, _ = nn.predict(X)
        score, _, _ = mnist_data.score_predictions(y, y_pred)
        cost = nn.cost(y)
        scores.append(score)
        costs.append(cost)

    def _print_training_time(self):
        """Prints the training time."""
        duration_text = _format_duration(self._total_duration)
        print("Train: {:s}".format(duration_text))

    def _plot_metric(self, ax, title, train, test):
        """Plots a single metric."""
        # Set up the axes.
        num_epochs = len(train)
        epochs = range(1, num_epochs + 1)
        ax.set_title(title)
        ax.set_xlim(0.5, num_epochs + 0.5)
        ax.set_xticks(epochs)

        # Plot the data.
        ax.plot(epochs, train, label="Train")
        ax.plot(epochs, test, label="Test")
        ax.legend()

####
# Neural networks
####

_UINT32_HIGH = 2 ** 32
_NUM_EPOCHS = 20

def _init_seeds(seed, start_epoch):
    """Randomly generates an initialization seed and a seed for each epoch."""
    np.random.seed(seed)
    init_seed = np.random.randint(_UINT32_HIGH)
    epoch_seeds = [np.random.randint(_UINT32_HIGH) for _ in range(_NUM_EPOCHS)]
    epoch_seeds = epoch_seeds[start_epoch:]
    return init_seed, epoch_seeds

def _init_nn(init_nn, start_epoch, seed):
    """Initializes the neural network."""
    np.random.seed(seed)
    nn = init_nn()
    if start_epoch:
        load_nn(nn, start_epoch)
    return nn

def _train_nn_epoch(nn, X, y, epoch, seed, epoch_fn):
    """Trains the neural network for an epoch."""
    np.random.seed(seed)
    minibatch_size = 32 * 2 ** (epoch // 4)
    start_time = time.time()
    nn.train(
        X, y,
        learning_rate=0.001, minibatch_size=minibatch_size, weight_decay=0.01)
    duration = time.time() - start_time
    _save_nn(nn, epoch + 1)
    if epoch_fn:
        epoch_fn(nn, duration)

def _save_best_epoch(nn, X, y):
    """Saves the best epoch from the final 4 epochs."""
    scores = [
        (_score_epoch(nn, X, y, epoch), epoch)
        for epoch in range(_NUM_EPOCHS - 3, _NUM_EPOCHS + 1)]
    best_epoch = max(scores)[1]
    load_nn(nn, best_epoch)
    _save_nn(nn)

def _score_epoch(nn, X, y, epoch):
    """Scores a single epoch."""
    load_nn(nn, epoch)
    y_pred, _ = nn.predict(X)
    score, _, _ = mnist_data.score_predictions(y, y_pred)
    return score

def _save_nn(nn, epoch=None):
    """Saves trained weights."""
    filename = _nn_filename(nn.name, epoch)
    nn.save(filename)

def _nn_filename(name, epoch):
    """Gets the filename for the trained weights."""
    if epoch is None:
        return "pretrain/{:s}.npz".format(name)
    else:
        return "pretrain/{:s}-epoch{:04d}.npz".format(name, epoch)

####
# Timing utilities
####

def _time_fn(name, fn):
    """Times a function."""
    start_time = time.time()
    ret = fn()
    duration = time.time() - start_time
    duration_text = _format_duration(duration)
    print("{:s}: {:s}".format(name, duration_text))
    return ret

def _format_duration(duration):
    """Formats a duration."""
    duration = int(1000 * duration)
    duration, ms = divmod(duration, 1000)
    duration, s = divmod(duration, 60)
    h, m = divmod(duration, 60)
    return "{:d}:{:02d}:{:02d}.{:03d}".format(h, m, s, ms)
