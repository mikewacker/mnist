import numpy as np

def dense(n_prev, n):
    """Creates a dense, fully-connected unit."""
    return _DenseUnit(n_prev, n)

def max_pool_2d(n_H_prev, n_W_prev, n_C_prev, *, pool_size, stride=0):
    """Creates a 2D max-pooling unit."""
    stride = stride or pool_size
    return _MaxPool2DUnit(n_H_prev, n_W_prev, n_C_prev, pool_size, stride)

def flatten(shape_in):
    """Creates a unit to flatten the inputs."""
    return _FlattenUnit(shape_in)

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

class _DenseUnit(_BaseUnit):
    """Dense, fully-connected unit."""

    def __init__(self, n_prev, n):
        """Initializes the unit."""
        shape_in = (n_prev,)
        shape_out = (n,)
        weights = _DenseUnit._init_weights(n_prev, n)
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

class _FlattenUnit(_BaseUnit):
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

class _MaxPool2DUnit(_BaseUnit):
    """2D max-pooling unit."""

    def __init__(self, n_H_prev, n_W_prev, n_C_prev, pool_size, stride):
        """Initializes the unit."""
        shape_in = (n_H_prev, n_W_prev, n_C_prev)
        shape_out = _MaxPool2DUnit._get_shape_out(
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

        # Iterate over each output position.
        n_H, n_W, _ = self.shape_out
        for h, w, h1_prev, h2_prev, w1_prev, w2_prev in _positions_2d(Z, f, s):

            # Pool a slice of input to the current position.
            A_prev_slice = A_prev[:, h1_prev:h2_prev, w1_prev:w2_prev, :]
            A_prev_slice_max = np.max(A_prev_slice, axis=(1, 2), keepdims=True)
            Z[:, h:h+1, w:w+1, :] = A_prev_slice_max

            # Set a slice of the mask.
            # Duplicate maxes are very unlikely and have minimal impact.
            mask_slice = mask[:, h1_prev:h2_prev, w1_prev:w2_prev, :]
            mask_slice[:] = (A_prev_slice - A_prev_slice_max) == 0.0

        # Cache and return.
        self._mask = mask
        return Z

    def backward(self, dZ, chained):
        """Back-propagates the output gradients."""
        if not chained:
            return None, ()
        f, s = self._f, self._s

        # Initialize.
        m = dZ.shape[0]
        dA_prev = np.empty((m, *self.shape_in))

        # Retrieve from cache.
        mask = self._mask

        # Iterate over each output position.
        n_H, n_W, _ = self.shape_out
        for h, w, h1_prev, h2_prev, w1_prev, w2_prev in _positions_2d(dZ, f, s):

            # Back-propagate from the current position to an input slice.
            dZ_slice = dZ[:, h:h+1, w:w+1, :]
            mask_slice = mask[:, h1_prev:h2_prev, w1_prev:w2_prev, :]
            dA_prev_slice = dA_prev[:, h1_prev:h2_prev, w1_prev:w2_prev, :]
            dA_prev_slice[:] = dZ_slice * mask_slice

        return dA_prev, ()

    @staticmethod
    def _get_shape_out(n_H_prev, n_W_prev, n_C_prev, pool_size, stride):
        """Gets the shape of the outputs."""
        n_H = _filter_size_out(n_H_prev, pool_size, stride, 0)
        n_W = _filter_size_out(n_W_prev, pool_size, stride, 0)
        return (n_H, n_W, n_C_prev)

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
