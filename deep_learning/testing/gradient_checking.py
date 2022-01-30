import itertools

import numpy as np

def check(cost_fn, grads_fn, *arrays):
    """Checks the computation of the gradients by approximating them.

    Recall the formula to approximate the derivative of a function.

    f'(x) = [f(x + eps) - f(x - eps)] / [(x + eps) - (x - eps)]
          = [f(x + eps) - f(x - eps)] / [2 * eps]

    The gradient approximation is essentially the multi-variable version,
    where "f" is a cost function that outputs a single value.

    Args:
        cost_fn: computes the cost from the numpy arrays
        grads_fn: computes the actuals gradients from the numpy arrays
        arrays: numpy arrays to run the check with

    Returns:
        diff: relative difference, which should be small (ideally, < 1e-7)
        grads: actual gradients
        grads_approx: approximated gradients
    """
    grads = grads_fn(*arrays)
    grads_approx = _approximate_gradients(cost_fn, arrays)
    diff = _compute_diff_arrays(grads, grads_approx)
    return diff, grads, grads_approx

def _approximate_gradients(cost_fn, arrays, epsilon=1e-7):
    """Approximates the gradients."""
    return tuple(
        _approximate_gradient(cost_fn, arrays, arrays_index, A, epsilon)
        for arrays_index, A in enumerate(arrays))

def _approximate_gradient(cost_fn, arrays, arrays_index, A, epsilon):
    """Approximates the gradient for a single array."""
    dA_approx = np.empty(A.shape)
    for index in _get_indices(A):
        deriv_approx = _approximate_derivative(
            cost_fn, arrays, arrays_index, index, epsilon)
        dA_approx[index] = deriv_approx
    return dA_approx

def _approximate_derivative(f, arrays, arrays_index, index, epsilon):
    """Approximates the derivative for a single element."""
    arrays_plus = _create_shifted_copy(arrays, arrays_index, index, epsilon)
    y_plus = f(*arrays_plus)
    arrays_minus = _create_shifted_copy(arrays, arrays_index, index, -epsilon)
    y_minus = f(*arrays_minus)
    deriv_approx = (y_plus - y_minus) / (2 * epsilon)
    return deriv_approx

def _get_indices(A):
    """Gets the index for each element in the array."""
    dim_ranges = [range(size) for size in A.shape]
    if len(dim_ranges) == 1:
        return dim_ranges[0]
    return itertools.product(*dim_ranges)

def _create_shifted_copy(arrays, arrays_index, index, delta):
    """Creates a copy of the arrays with one element shifted."""
    A = arrays[arrays_index]
    A_shifted = A.copy()
    A_shifted[index] += delta
    arrays_shifted = list(arrays)
    arrays_shifted[arrays_index] = A_shifted
    return arrays_shifted

def _compute_diff_arrays(arrays1, arrays2):
    """Computes the difference between two tuples of arrays"""
    vector1 = _flatten(arrays1)
    vector2 = _flatten(arrays2)
    return _compute_diff_flat(vector1, vector2)

def _compute_diff_flat(vector1, vector2):
    """Computes the difference between two flat vectors"""
    diff = np.linalg.norm(vector2 - vector1)
    norm = np.linalg.norm(vector1) + np.linalg.norm(vector2)
    diff_normed = diff / norm
    return diff_normed

def _flatten(arrays):
    """Flattens the arrays into a single vector."""
    splices = _get_flat_splices(arrays)
    n = splices[-1][1]
    vector = np.empty(n)
    for (offset, end), A in zip(splices, arrays):
        vector[offset:end] = A.reshape(-1)
    return vector

def _get_flat_splices(arrays):
    """Gets each array's splice in the flat array."""
    splices = []
    offset = 0
    for A in arrays:
        size = np.prod(A.shape)
        end = offset + size
        splice = (offset, end)
        splices.append(splice)
        offset = end
    return splices
