import numpy as np

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
