import numpy as np

def dense(n_prev, n):
    """Creates a dense, fully-connected unit."""
    return _Dense(n_prev, n)

def convolution_2d(
    n_H_prev, n_W_prev, n_C_prev, n_C, *, kernel_size, stride=1, padding=0):
    """Creates a 2D convulational unit."""
    return _Convolution2D(
        n_H_prev, n_W_prev, n_C_prev, n_C, kernel_size, stride, padding)

def max_pool_2d(n_H_prev, n_W_prev, n_C_prev, *, pool_size, stride=0):
    """Creates a 2D max-pooling unit."""
    stride = stride or pool_size
    return _MaxPool2D(n_H_prev, n_W_prev, n_C_prev, pool_size, stride)

def flatten(shape_in):
    """Creates a unit to flatten the inputs."""
    return _Flatten(shape_in)

"""
A unit extends _BaseUnit. It must invoke the following super-constructor:

*   super().__init__(shape_in, shape_out, weights=())

Notes:

*   Corresponding properties exist that can be used in your implementation.
*   shape_in, shape_out exclude the first dimension for the number of samples.
*   weights is a tuple of numpy arrays.

A unit must also implement the following methods:

*   forward(A_prev)
*   backward(dZ, chained)

Notes:

*   backward() returns the input gradients and a tuple of weight gradients.
*   backward() doesn't calculate the input gradients if chained is False.
"""

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

####
# Units
####

class _Dense(_BaseUnit):
    """Dense, fully-connected unit."""

    def __init__(self, n_prev, n):
        """Initializes the unit."""
        shape_in = (n_prev,)
        shape_out = (n,)
        weights = _Dense._init_weights(n_prev, n)
        super().__init__(shape_in, shape_out, weights)

    def forward(self, A_prev):
        """Feeds the inputs forward."""
        W, b = self.weights

        # Feed forward.
        Z = A_prev @ W + b

        # Cache and return.
        self._A_prev = A_prev
        return Z

    def backward(self, dZ, chained):
        """Back-propagates the output gradients."""
        W, _ = self.weights

        # Retrieve from cache.
        A_prev = self._A_prev

        # Back-propagate to weights.
        dW = A_prev.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)

        # Back-propagate to inputs.
        dA_prev = dZ @ W.T if chained else None

        return dA_prev, (dW, db)

    @staticmethod
    def _init_weights(n_prev, n):
        """Initializes the weights."""
        W_shape = (n_prev, n)
        W = _glorot_normal_initialization(W_shape, n_prev, n)
        b = np.zeros((1, n))
        return W, b

class _Convolution2D(_BaseUnit):
    """2D convolutional unit."""

    def __init__(
        self, n_H_prev, n_W_prev, n_C_prev, n_C, kernel_size, stride, padding):
        """Initializes the unit."""
        shape_in = (n_H_prev, n_W_prev, n_C_prev)
        shape_out = _Convolution2D._get_shape_out(
            n_H_prev, n_W_prev, n_C, kernel_size, stride, padding)
        weights = _Convolution2D._init_weights(n_C_prev, n_C, kernel_size)
        super().__init__(shape_in, shape_out, weights)

        self._f = kernel_size
        self._s = stride
        self._p = padding

    def forward(self, A_prev):
        """Feeds the inputs forward."""
        f, s, p = self._f, self._s, self._p
        W, b = self.weights

        # Initialize.
        m = A_prev.shape[0]
        Z = np.empty((m, *self.shape_out))

        # Pad the input.
        if p > 0:
            pad_width = ((0, 0), (p, p), (p, p), (0, 0))
            A_prev = np.pad(A_prev, pad_width, "constant", constant_values=0)

        # Grab slices of the input and output.
        for h, w, h1_prev, h2_prev, w1_prev, w2_prev in _positions_2d(Z, f, s):
            A_prev_slice = A_prev[:, h1_prev:h2_prev, w1_prev:w2_prev, :]
            Z_slice = Z[:, h, w, :]

            # Computer the convolution for the input slice.
            A_prev_slice = A_prev_slice.reshape((*A_prev_slice.shape, 1))
            Z_slice[:] = np.sum(W * A_prev_slice, axis=(1,2,3)) + b

        # Cache and return.
        self._A_prev = A_prev
        return Z

    def backward(self, dZ, chained):
        """Back-propagates the output gradients."""
        f, s, p = self._f, self._s, self._p
        W, b = self.weights

        # Retrieve from cache.
        A_prev = self._A_prev

        # Initialize.
        m, _, _, n_C = dZ.shape
        dW = np.zeros(W.shape)
        db = np.zeros(b.shape)
        dA_prev = np.zeros(A_prev.shape) if chained else None

        # Grab slices of the output and input.
        for h, w, h1_prev, h2_prev, w1_prev, w2_prev in _positions_2d(dZ, f, s):
            dZ_slice = dZ[:, h, w, :]
            A_prev_slice = A_prev[:, h1_prev:h2_prev, w1_prev:w2_prev, :]

            # Back-propagate to the weights.
            dZ_slice = dZ_slice.reshape(m, 1, 1, 1, n_C)
            A_prev_slice = A_prev_slice.reshape((*A_prev_slice.shape, 1))
            dW += np.sum(A_prev_slice * dZ_slice, axis=0)
            db += np.sum(dZ_slice, axis=(0, 1, 2, 3))

            # Back-propagate to the input slice.
            if not chained:
                continue
            dA_prev_slice = dA_prev[:, h1_prev:h2_prev, w1_prev:w2_prev, :]
            dA_prev_slice += np.sum(W * dZ_slice, axis=4)

        # Remove padding from the input.
        if chained and p > 0:
            dA_prev = dA_prev[:, p:-p, p:-p, :]

        return dA_prev, (dW, db)

    @staticmethod
    def _get_shape_out(n_H_prev, n_W_prev, n_C, kernel_size, stride, padding):
        """Gets the shape of the output."""
        n_H = _filter_size_out(n_H_prev, kernel_size, stride, padding)
        n_W = _filter_size_out(n_W_prev, kernel_size, stride, padding)
        return (n_H, n_W, n_C)

    @staticmethod
    def _init_weights(n_C_prev, n_C, kernel_size):
        """Initializes the weights using Xavier initialization."""
        W_shape = (kernel_size, kernel_size, n_C_prev, n_C)
        fan_in = kernel_size * kernel_size * n_C_prev
        fan_out = n_C
        W = _glorot_normal_initialization(W_shape, fan_in, fan_out)
        b = np.zeros(n_C)
        return W, b

class _MaxPool2D(_BaseUnit):
    """2D max-pooling unit."""

    def __init__(self, n_H_prev, n_W_prev, n_C_prev, pool_size, stride):
        """Initializes the unit."""
        shape_in = (n_H_prev, n_W_prev, n_C_prev)
        shape_out = _MaxPool2D._get_shape_out(
            n_H_prev, n_W_prev, n_C_prev, pool_size, stride)
        super().__init__(shape_in, shape_out)

        self._f = pool_size
        self._s = stride

    def forward(self, A_prev):
        """Feeds the inputs forward."""
        f, s = self._f, self._s

        # Initialize.
        m = A_prev.shape[0]
        mask = np.empty(A_prev.shape, dtype=bool)
        Z = np.empty((m, *self.shape_out))

        # Grab slices of the input, mask, and output.
        for h, w, h1_prev, h2_prev, w1_prev, w2_prev in _positions_2d(Z, f, s):
            A_prev_slice = A_prev[:, h1_prev:h2_prev, w1_prev:w2_prev, :]
            mask_slice = mask[:, h1_prev:h2_prev, w1_prev:w2_prev, :]
            Z_slice = Z[:, h:h+1, w:w+1, :]

            # Pool the input slice.
            A_prev_slice_max = np.max(A_prev_slice, axis=(1, 2), keepdims=True)
            Z_slice[:] = A_prev_slice_max

            # Set a slice of the mask.
            # Duplicate maxes are very unlikely and have minimal impact.
            mask_slice[:] = (A_prev_slice - A_prev_slice_max) == 0.0

        # Cache and return.
        self._mask = mask
        return Z

    def backward(self, dZ, chained):
        """Back-propagates the output gradients."""
        if not chained:
            return None, ()
        f, s = self._f, self._s

        # Retrieve from cache.
        mask = self._mask

        # Initialize.
        dA_prev = np.empty(mask.shape)

        # Grab slices of the output, mask, and input.
        for h, w, h1_prev, h2_prev, w1_prev, w2_prev in _positions_2d(dZ, f, s):
            dZ_slice = dZ[:, h:h+1, w:w+1, :]
            mask_slice = mask[:, h1_prev:h2_prev, w1_prev:w2_prev, :]
            dA_prev_slice = dA_prev[:, h1_prev:h2_prev, w1_prev:w2_prev, :]

            # Back-propagate from the current position to the input slice.
            dA_prev_slice[:] = dZ_slice * mask_slice

        return dA_prev, ()

    @staticmethod
    def _get_shape_out(n_H_prev, n_W_prev, n_C_prev, pool_size, stride):
        """Gets the shape of the outputs."""
        n_H = _filter_size_out(n_H_prev, pool_size, stride, 0)
        n_W = _filter_size_out(n_W_prev, pool_size, stride, 0)
        return (n_H, n_W, n_C_prev)

class _Flatten(_BaseUnit):
    """Flattens the inputs."""

    def __init__(self, shape_in):
        """Initializes the unit."""
        n = np.prod(np.array(shape_in))
        shape_out = (n,)
        super().__init__(shape_in, shape_out)

    def forward(self, A_prev):
        """Feeds the inputs forward."""
        Z = A_prev.reshape((-1, *self.shape_out))
        return Z

    def backward(self, dZ, chained):
        """Back-propagates the output gradients."""
        if not chained:
            return None, ()

        dA_prev = dZ.reshape((-1, *self.shape_in))
        return dA_prev, ()

def _filter_size_out(n_prev, filter_size, stride, padding):
    """Calculates the output size for a filter."""
    return ((n_prev + 2 * padding - filter_size) // stride) + 1

def _positions_2d(Z, filter_size, stride):
    """Generates each output position and the corresponding input slice."""
    _, n_H, n_W, _ = Z.shape
    for h in range(n_H):
        h1_prev = h * stride
        h2_prev = h1_prev + filter_size
        for w in range(n_W):
            w1_prev = w * stride
            w2_prev = w1_prev + filter_size
            yield h, w, h1_prev, h2_prev, w1_prev, w2_prev

####
# Weight initialization
####

def _glorot_normal_initialization(shape, fan_in, fan_out):
    """Initializes weights using Glorot normal initialization."""
    std = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(0, std, shape)
