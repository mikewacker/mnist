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

def binary_output():
    """Creates a binary output."""
    return _Output(2)

def multiclass_output(C):
    """Creates a multiclass output with the specified number of classes."""
    if C <= 2:
        msg = "Multiclass output must have at least 3 classes: {:d}".format(C)
        raise ValueError(msg)
    return _Output(C)

"""
An activation must implement the following methods:

*   forward(Z)
*   backward(dA)

An output is an activation that must also implement the following properties:

*   is_multiclass [read-only]
*   C [read-only]
*   Y [read-write]
*   cost [read-only]
*   pred [read-only]
*   prob [read-only]

Notes:

*   Y's shape is (m, 1), (m, C) respectively for binary and multiclass outputs.
*   prob's shape is (m,), (m, C) respectively for binary and multiclass outputs.
*   Y must be set to call cost or backward().
*   The argument for backward() is ignored for an output.
"""

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

###
# Output
###

class _Output(object):
    """Output using sigmoid/softmax activation with cross-entropy cost."""

    def __init__(self, C):
        """Initializes the output with the number of classes."""
        self._C = C
        self._Y = None

    @property
    def is_multiclass(self):
        """Determines if the output is a multiclass output."""
        return self._C > 2

    @property
    def C(self):
        """Gets the number of classes."""
        return self._C

    @property
    def Y(self):
        """Gets the true labels."""
        if self._Y is None:
            raise RuntimeError("Y must be set")
        return self._Y

    @Y.setter
    def Y(self, value):
        """Sets the true labels."""
        self._Y = value

    def forward(self, Z):
        """Feeds the inputs forward."""
        # Feed forward.
        if self.is_multiclass:
            A = _softmax(Z)
        else:
            A = _sigmoid(Z)

        # Cache and return.
        self._A = A
        return A

    @property
    def cost(self):
        """Gets the cost for the predictions."""
        # Retrieve from cache.
        A = self._A
        Y = self.Y

        # Calculate the cost.
        if self.is_multiclass:
            prob = np.sum(Y * A, axis=1)
        else:
            prob = Y * A + (1 - Y) * (1 - A)
            prob = prob.reshape(-1)
        loss = -np.log(prob)
        cost = np.mean(loss)
        return cost

    @property
    def pred(self):
        """Gets the predicted labels."""
        # Retrieve from cache.
        A = self._A

        # Calculate the predictions.
        if self.is_multiclass:
            pred = np.argmax(A, axis=1)
        else:
            pred = (A > 0.5).astype(int)
            pred = pred.reshape(-1)
        return pred

    @property
    def prob(self):
        """Gets the predicted probabilities."""
        # Retrieve from cache.
        A = self._A

        # Calculate the probabilities.
        if self.is_multiclass:
            prob = A
        else:
            prob = A.reshape(-1)
        return prob

    def backward(self, _):
        """Back-propagates the output gradients."""
        # Retrieve from cache.
        A = self._A
        Y = self.Y

        # Back-propagate.
        m = Y.shape[0]
        dZ = (A - Y) / m
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
