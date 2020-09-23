import numpy as np

def activation(fn):
    """Creates an activation using the provided function."""
    if fn == "relu":
        return _ReluActivation()
    if fn == "sigmoid":
        return _SigmoidActivation()
    if fn == "tanh":
        return _TanhActivation()
    msg = "unknown activation function: {}".format(fn)
    raise ValueError(msg)

####
# Activations
####

class _ReluActivation(object):
    """ReLU activation."""

    def forward(self, Z):
        """Feeds the inputs forward."""
        # Feed forward.
        mask = Z >= 0
        A = Z * mask

        # Cache and return.
        self._mask = mask
        return A

    def backward(self, dA):
        """Back-propagates the output gradients."""
        # Retrieve from cache.
        mask = self._mask

        # Back-propagate.
        dZ = dA * mask
        return dZ

class _SigmoidActivation(object):
    """Sigmoid activation."""

    def forward(self, Z):
        """Feeds the inputs forward."""
        # Feed forward.
        A = _sigmoid(Z)

        # Cache and return.
        self._A = A
        return A

    def backward(self, dA):
        """Back-propagates the output gradients."""
        # Retrieve from cache.
        A = self._A

        # Back-propagate.
        dZ = dA * A * (1 - A)
        return dZ

class _TanhActivation(object):
    """Hyperbolic tangent activation."""

    def forward(self, Z):
        """Feeds the inputs forward."""
        # Feed forward.
        A = np.tanh(Z)

        # Cache and return.
        self._A = A
        return A

    def backward(self, dA):
        """Back-propagates the output gradients."""
        # Retrieve from cache.
        A = self._A

        # Back-propagate.
        dZ = dA * (1 - np.square(A))
        return dZ

####
# Activation functions
####

def _sigmoid(Z):
    """Applies sigmoid activation to each element."""
    A = 1 / (1 + np.exp(-Z))
    return A

def _softmax(Z):
    """Applies softmax activation to each row."""
    # Softmax is invariant to shifts.
    # For computational stability, shift each row so that the max value is 0.
    Z_max = np.max(Z, axis=1, keepdims=True)
    Z_shift = Z - Z_max

    # Apply softmax activation.
    T = np.exp(Z_shift)
    T_norm = np.sum(T, axis=1, keepdims=True)
    A = T / T_norm
    return A
