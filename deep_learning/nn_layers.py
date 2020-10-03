import numpy as np

from . import _nn_units
from . import _nn_activations

def dense(n_prev, n, *, activation="relu"):
    """Creates a dense, fully-connected layer.

    Args:
        n_prev: number of inputs from the previous layer
        n: number of nodes for this layer
        activation: activation function for this layer,
            one of {sigmoid, tanh, relu}
    """
    unit = _nn_units.dense(n_prev, n)
    activation = _nn_activations.activation(activation)
    return _ActivatedLayer(unit, activation)

def convolution_2d(
        n_H_prev, n_W_prev, n_C_prev, n_C, *,
        kernel_size, stride=1, padding=0, activation="relu"):
    """Creates a 2D convulational layer.

    Args:
        n_H_prev: height of inputs
        n_W_prev: width of inputs
        n_C_prev: number of input channels
        n_C: number of output channels
        kernel_size: size of the kernel
        stride: stride length for the kernel
        padding: size of pad to add to each side
        activation: activation function for this layer,
            one of {sigmoid, tanh, relu}
    """
    unit = _nn_units.convolution_2d(
        n_H_prev, n_W_prev, n_C_prev, n_C,
        kernel_size=kernel_size, stride=stride, padding=padding)
    activation = _nn_activations.activation(activation)
    return _ActivatedLayer(unit, activation)

def max_pool_2d(
        n_H_prev, n_W_prev, n_C_prev, *, pool_size, stride=0):
    """Creats a 2D max-pooling layer.

    Args:
        n_H_prev: height of inputs
        n_W_prev: width of inputs
        n_C_prev: number of input channels
        pool_size: size of the pool
        stride: stride length for the pool, or 0 to match the pool size
    """
    return _nn_units.max_pool_2d(
        n_H_prev, n_W_prev, n_C_prev, pool_size=pool_size, stride=stride)

def flatten(shape_in):
    """Creates a layer to flatten the inputs.

    Args:
        shape_in: input shape, excluding the number of samples
    """
    return _nn_units.flatten(shape_in)

def binary_output(n_prev):
    """Creates a binary output layer.

    Args:
        n_prev: number of inputs from the previous layer
    """
    unit = _nn_units.dense(n_prev, 1)
    output = _nn_activations.binary_output()
    return _OutputLayer(unit, output)

def multiclass_output(n_prev, C):
    """Creates a multi-class output layer.

    Args:
        n_prev: number of inputs from the previous layer
        C: number of output classes
    """
    unit = _nn_units.dense(n_prev, C)
    output = _nn_activations.multiclass_output(C)
    return _OutputLayer(unit, output)

"""
Hidden layers can be created in one of two ways:

*   using a unit by itself
*   composing a unit and an activation via _ActivatedLayer

An output layer can be created in one way:

*   composing a unit and an output via _OutputLayer
"""

####
# Layer shapes
####

def check_layer_shapes(layers):
    """Checks the shapes of the layers."""
    prev_layer = layers[0]
    for index, layer in enumerate(layers[1:], 1):
        if prev_layer.shape_out == layer.shape_in:
            prev_layer = layer
            continue
        msg = "layer {:d} expects shape {:s}, got {:s} from layer {:d}".format(
            index + 1, _shape_text(layer.shape_in),
            _shape_text(prev_layer.shape_out), index)
        raise ValueError(msg)

def check_input_shape(X, layer):
    """Checks the shape of the inputs."""
    shape_in = X.shape[1:]
    if shape_in == layer.shape_in:
        return
    msg = "layer 1 expects shape {:s}, got {:s} from input".format(
        _shape_text(layer.shape_in), _shape_text(shape_in))
    raise ValueError(msg)

def onehot_output(y, output):
    """One-hot encodes the output labels."""
    if not output.is_multiclass:
        return y.reshape((-1, 1))
    m = y.shape[0]
    Y = np.zeros((m, output.C))
    Y[np.arange(m), y] = 1
    return Y

def _shape_text(shape):
  """Represents the shape as text."""
  sizes = ["m"] + [format(n, "d") for n in shape]
  return "({:s})".format(", ".join(sizes))

####
# Layer composition
####

class _ActivatedLayer(object):
    """Builds a hidden layer from a unit an activation function."""

    def __init__(self, unit, activation):
        """Initializes the layer."""
        self._unit = unit
        self._activation = activation

    @property
    def shape_in(self):
        """Gets the input shape, excluding the number of samples."""
        return self._unit.shape_in

    @property
    def shape_out(self):
        """Gets the output shape, excluding the number of samples."""
        return self._unit.shape_out

    @property
    def num_params(self):
        """Gets the number of trainable parameters."""
        return self._unit.num_params

    @property
    def weights(self):
        """Gets the weights."""
        return self._unit.weights

    @weights.setter
    def weights(self, value):
        """Sets the weights."""
        self._unit.weights = value

    def load_weights(self, nn_dict, index):
        """Loads pre-trained weights from a dictionary."""
        self._unit.load_weights(nn_dict, index)

    def save_weights(self, nn_dict, index):
        """Saves trained weights to a dictionary."""
        self._unit.save_weights(nn_dict, index)

    def forward(self, A_prev):
        """Feeds the inputs forward."""
        Z = self._unit.forward(A_prev)
        return self._activation.forward(Z)

    def backward(self, dA, chained):
        """Back-propagates the output gradients."""
        dZ = self._activation.backward(dA)
        return self._unit.backward(dZ, chained)

class _OutputLayer(_ActivatedLayer):
    """Builds an output layer from a unit and output."""

    def __init__(self, unit, output):
        """Initializes the layer."""
        super().__init__(unit, output)
        self._output = output

    @property
    def is_multiclass(self):
        """Determines if the output is a multiclass output."""
        return self._output.is_multiclass

    @property
    def C(self):
        """Gets the number of classes."""
        return self._output.C

    @property
    def Y(self):
        """Gets the true labels."""
        return self._output.Y

    @Y.setter
    def Y(self, value):
        """Sets the true labels."""
        self._output.Y = value

    @property
    def cost(self):
        """Gets the cost for the predictions."""
        return self._output.cost

    @property
    def pred(self):
        """Gets the predicted labels."""
        return self._output.pred

    @property
    def prob(self):
        """Gets the predicted probabilities."""
        return self._output.prob
