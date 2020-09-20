import numpy as np

def gradient_descent():
    """Creates an optimizer using gradient descent."""
    return _Optimizer("gradient_descent", _GradientDescent)

def adam(*, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Creates an optimizer using Adam.

    Args:
        beta1: exponential decay rate for the first moment estimates
        beta2: exponential decay rate for the second moment estimates
        epsilon: small constant to prevent division by zero
    """
    return _Optimizer("adam", _Adam, beta1, beta2, epsilon)

"""
To create a new optimizer, you only need to implement a new stepper.
A stepper must implement the following properties and methods:

*   __init__(shape, *args)
*   state [read-write]
*   step(dW)

Notes:

*   One stepper will be created for each numpy array of weights.
*   state is a tuple of numpy arrays that is used for persistence.
*   The optimizer will invoke step() and scale each step by the learning rate.

To create an optimizer, you just need to invoke its constructor:

*   _Optimizer(name, stepper_type, *args)

Additional args will be passed through to the stepper's constructor.
The optimizer will create one stepper for each numpy array of weights.
"""

####
# Optimizer
####

class _Optimizer(object):
    """Optimizes weights for all layers."""

    def __init__(self, name, stepper_type, *args):
        """Initializers the optimizer."""
        self._name = name
        self._stepper_type = stepper_type
        self._args = args

    def init_steppers(self, layers):
        """Initializes the steppers for all layers."""
        self._all_steppers = [
            self._init_layer_steppers(layer)
            for layer in layers]

    def load_state(self, nn_dict):
        """Loads the state from a dictionary."""
        for index, layer_steppers in enumerate(self._all_steppers):
            for W_index, stepper in enumerate(layer_steppers):
                self._load_stepper_state(nn_dict, index, W_index, stepper)

    def save_state(self, nn_dict):
        """Saves the state to a dictionary."""
        for index, layer_steppers in enumerate(self._all_steppers):
            for W_index, stepper in enumerate(layer_steppers):
                self._save_stepper_state(nn_dict, index, W_index, stepper)

    def update_weights(self, index, layer, grads, learning_rate):
        """Updates the weights for a layer."""
        steppers = self._all_steppers[index]
        weights = []
        for stepper, W, dW in zip(steppers, layer.weights, grads):
            step = stepper.step(dW)
            W = W - learning_rate * step
            weights.append(W)
        layer.weights = tuple(weights)

    def _init_layer_steppers(self, layer):
        """Initializes the steppers for a layer."""
        return [
            self._stepper_type(W.shape, *self._args)
            for W in layer.weights]

    def _load_stepper_state(self, nn_dict, index, W_index, stepper):
        """Loads the state for a stepper from a dictionary."""
        state = []
        for S_index, _ in enumerate(stepper.state):
            key = self._state_key(index, W_index, S_index)
            S = nn_dict.get(key, None)
            self._check_loaded_S(key, S)
            state.append(S)
        stepper.state = tuple(state)

    def _check_loaded_S(self, key, S):
        """Checks the state that was loaded."""
        if S is None:
            msg = "{:s} not found".format(key)
            raise ValueError(msg)

    def _save_stepper_state(self, nn_dict, index, W_index, stepper):
        """Saves the state for a stepper to a dictionary."""
        for S_index, S in enumerate(stepper.state):
            key = self._state_key(index, W_index, S_index)
            nn_dict[key] = S

    def _state_key(self, index, W_index, S_index):
        """Gets the dictionary key for the state."""
        return "layer {:d}: optimizer-{:s}[{:d}][{:d}]".format(
            index + 1, self._name, W_index, S_index)

####
# Steppers
####

class _GradientDescent(object):
    """Takes steps using gradient descent."""

    def __init__(self, shape):
        """Initializes the stepper."""
        self.state = ()

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
