import numpy as np

####
# Steppers
####

class _GradientDescent(object):
    """Takes steps using gradient descent."""

    def __init__(self, shape):
        """Initializes the stepper."""
        pass

    @property
    def state(self):
        """Gets the state."""
        return ()

    @state.setter
    def state(self, _):
        """Sets the state."""
        pass

    def step(self, dW):
        """Takes a step."""
        return dW

class _Adam(object):
    """Takes steps using Adam."""

    def __init__(self, shape, beta1, beta2, epsilon):
        """Initializes the stepper."""
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._vdW = np.zeros(shape)
        self._sdW = np.zeros(shape)
        self._t = 0

    @property
    def state(self):
        """Gets the state."""
        t = np.array([self._t])
        return (self._vdW, self._sdW, t)

    @state.setter
    def state(self, value):
        """Sets the state."""
        self._vdW, self._sdW, t = value
        self._t = t[0]

    def step(self, dW):
        """Takes a step."""
        # Retrieve from cache.
        beta1, beta2, epsilon = self._beta1, self._beta2, self._epsilon
        vdW = self._vdW
        sdW = self._sdW
        t = self._t

        # Calculate vdW, sdW with bias correction.
        t += 1
        vdW = beta1 * vdW + (1 - beta1) * dW
        vdW_bc = _bias_correction(vdW, beta1, t)
        sdW = beta2 * sdW + (1 - beta2) * np.square(dW)
        sdW_bc = _bias_correction(sdW, beta2, t)

        # Compute the step.
        step = vdW_bc / (np.sqrt(sdW_bc) + epsilon)

        # Cache and return.
        self._vdW = vdW
        self._sdW = sdW
        self._t = t
        return step

def _bias_correction(X, beta, t):
    """Performs bias correction."""
    bc = 1 - beta ** t
    return X / bc
