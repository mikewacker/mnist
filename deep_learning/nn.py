import numpy as np

from . import _nn_minibatch
from . import nn_layers

class NeuralNetwork(object):
    """Neural network for classification."""

    def __init__(
        self, *, hidden_layers, output_layer, optimizer, preprocess_fn=None):
        """Initalizes the neural network.

        Args:
            hidden_layers: list of hidden layers
            output_layer: output layer
            optimizer: optimizer that updates weights
            preprocess_fn: function to preprocess the inputs
        """
        layers = hidden_layers + [output_layer]
        nn_layers.check_layer_shapes(layers)
        optimizer.init_steppers(layers)

        self._preprocess_fn = preprocess_fn
        self._layers = layers
        self._output = output_layer
        self._optimizer = optimizer

    def load(self, file):
        """Loads pre-trained weights from a file.

        Args:
            file: filename or a file-like object
        """
        nn_dict = np.load(file)
        for index, layer in enumerate(self._layers):
            layer.load_weights(nn_dict, index)
        self._optimizer.load_state(nn_dict)

    def save(self, file):
        """Saves trained weights to a file.

        Args:
            file: filename or a file-like object
        """
        nn_dict = {}
        for index, layer in enumerate(self._layers):
            layer.save_weights(nn_dict, index)
        self._optimizer.save_state(nn_dict)
        np.savez_compressed(file, **nn_dict)

    def train(
        self, X, y, *,
        learning_rate, minibatch_size, num_epochs=1, weight_decay=0,
        progress_fn=None):
        """Trains the network.

        Args:
            X: inputs as a (m, ...) numpy array
            y: true labels as a (m,) numpy array
            learning_rate: learning rate
            minibatch_size: mini-batch size, or 0 to use the entire batch
            num_epochs: number of epochs to train
            weight_decay: weight decay to apply at each step
            progress_fn: function invoked with the cost after each iteration
        """
        X = self._preprocess_input(X)
        Y = nn_layers.onehot_output(y, self._output)
        for _ in range(num_epochs):
            for X_mb, Y_mb in _nn_minibatch.minibatches(X, Y, minibatch_size):
                self._train_minibatch(
                    X_mb, Y_mb, learning_rate, weight_decay, progress_fn)

    def predict(self, X):
        """Predicts the output labels.

        Args:
            X: inputs as a (m, ...) numpy array

        Returns:
            y_pred: predicted labels as a (m,) numpy array
            Y_prob: predicted probabilities as a (m,) or (m, C) numpy array,
                depending on whether the outputs are binary or multi-class
        """
        X = self._preprocess_input(X)
        return self._forward_predict(X)

    def cost(self, y):
        """Gets the cost used to evaluate the previous predictions.

        Args:
            y: true labels as a (m,) numpy array

        Returns:
            cost used to evaluate the previous predictions
        """
        Y = nn_layers.onehot_output(y, self._output)
        self._output.Y = Y
        return self._output.cost

    def _preprocess_input(self, X):
        """Preprocesses the inputs."""
        if self._preprocess_fn:
            X = self._preprocess_fn(X)
        nn_layers.check_input_shape(X, self._layers[0])
        return X

    def _train_minibatch(self, X, Y, learning_rate, weight_decay, progress_fn):
        """Trains a single mini-batch."""
        self._forward_train(X, Y)
        if progress_fn:
            progress_fn(self._output.cost)
        self._backward(learning_rate, weight_decay)

    def _forward_train(self, A, Y):
        """Feeds the inputs forward for training."""
        self._output.Y = Y
        for layer in self._layers:
            A = layer.forward(A)

    def _backward(self, learning_rate, weight_decay):
        """Performs back-propagation, also updating weights."""
        dA = None
        for index, layer in reversed(list(enumerate(self._layers))):
            chained = index > 0
            dA, grads = layer.backward(dA, chained)
            self._optimizer.update_weights(
                index, layer, grads, learning_rate, weight_decay)

    def _forward_predict(self, A):
        """Feeds the inputs forward for predicting."""
        self._output.Y = None
        for layer in self._layers:
            A = layer.forward(A)
        return self._output.pred, self._output.prob
