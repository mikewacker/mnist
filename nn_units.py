import numpy as np

####
# Base unit
####

class _BaseUnit(object):

    def __init__(self, shape_in, shape_out, weights=()):
        """Initializes the unit."""
        if type(weights) != tuple:
            raise ValueError("weights must be a tuple of numpy arrays")
        self._shape_in = (shape_in,) if np.isscalar(shape_in) else shape_in
        self._shape_out = (shape_out,) if np.isscalar(shape_out) else shape_out
        self._weights = weights

    @property
    def shape_in(self):
        """Gets the input shape, excluding the number of samples."""
        return self._shape_in

    @property
    def shape_out(self):
        """Gets the output shape, excluding the number of samples."""
        return self._shape_out

    @property
    def num_params(self):
        """Gets the number of parameters."""
        return sum(
            np.prod(np.array(W.shape))
            for W in self.weights)

    @property
    def weights(self):
        """Gets the weights."""
        return self._weights

    @weights.setter
    def weights(self, value):
        """Sets the weights."""
        if type(value) != tuple:
            raise ValueError("weights must be a tuple of numpy arrays")
        if len(value) != len(self.weights):
            msg = "weights expected {:d} arrays, got {:d}".format(
                len(self.weights), len(value))
            raise ValueError(msg)
        for W_index, (W, W_prev) in enumerate(zip(value, self.weights)):
            name = self._weights_name(None, W_index)
            self._check_W_shape(name, W, W_prev)
        self._weights = value

    def load_weights(self, nn_dict, index):
        """Loads pre-trained weights from a dictionary."""
        self._weights = tuple(
            self._load_W(nn_dict, index, W_index, W_prev)
            for W_index, W_prev in enumerate(self.weights))

    def save_weights(self, nn_dict, index):
        """Saves trained weights to a dictionary."""
        nn_dict.update({
            self._weights_name(index, W_index): W
            for W_index, W in enumerate(self.weights)})

    def _load_W(self, nn_dict, index, W_index, W_prev):
        """Loads a single array of weights from a dictionary."""
        name = self._weights_name(index, W_index)
        W = nn_dict.get(name, None)
        self._check_W_shape(name, W, W_prev)
        return W

    def _weights_name(self, index, W_index):
        """Gets the name for the weights."""
        if index is None:
            name = "weights[{:d}]".format(W_index)
        else:
            name = "layer {:d}: weights[{:d}]".format(index + 1, W_index)
        return name

    def _check_W_shape(self, name, W, W_prev):
        """Checks the shape of W."""
        if W is None:
            msg = "{:s} not found".format(name)
            raise ValueError(msg)
        if W.shape != W_prev.shape:
            msg = "{:s} expects shape {}, got {}".format(
                name, W_prev.shape, W.shape)
            raise ValueError(msg)
