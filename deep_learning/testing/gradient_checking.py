import itertools

import numpy as np

def check(cost_fn, grads_fn, *arrays):
    """Checks the computation of the gradients by approximating them.

    Args:
        cost_fn: computes the cost from the numpy arrays
        grads_fn: computes the gradient for each numpy array
        arrays: numpy arrays to run the check with

    Returns:
        diff: relative difference, which should be a small value
        grads: computed gradients
        grads_approx: approximated gradients, using the cost function
    """
    grads = grads_fn(*arrays)
    grads_approx = _approximate_gradients(cost_fn, arrays)
    diff = _compute_diff(grads, grads_approx)
    return diff, grads, grads_approx

def _approximate_gradients(cost_fn, arrays, epsilon=1e-7):
    """Approximates the gradients."""
    return tuple(
        _approximate_gradient(cost_fn, arrays, A_index, A, epsilon)
        for A_index, A in enumerate(arrays))

def _approximate_gradient(cost_fn, arrays, A_index, A, epsilon):
    """Approximates the gradient for an array."""
    dA_approx = np.empty(A.shape)
    for index in _get_indices(A):
        deriv_approx = _approximate_derivative(
            cost_fn, arrays, A_index, A, index, epsilon)
        dA_approx[index] = deriv_approx
    return dA_approx

def _get_indices(A):
    """Gets the indices for an array."""
    dim_ranges = [range(size) for size in A.shape]
    if len(dim_ranges) == 1:
        return dim_ranges[0]
    return itertools.product(*dim_ranges)

def _approximate_derivative(f, arrays, A_index, A, index, epsilon):
    """Approximates the derivative for a single element."""
    A_plus = _shift_element(A, index, epsilon)
    arrays_plus = _merge_arrays(arrays, A_index, A_plus)
    y_plus = f(*arrays_plus)
    A_minus = _shift_element(A, index, -epsilon)
    arrays_minus = _merge_arrays(arrays, A_index, A_minus)
    y_minus = f(*arrays_minus)
    deriv_approx = (y_plus - y_minus) / (2 * epsilon)
    return deriv_approx

def _shift_element(A, index, delta):
    """Shifts a single element in the array."""
    A_shift = A.copy()
    A_shift[index] += delta
    return A_shift

def _merge_arrays(arrays, A_index, A):
    """Merges a changed array back into the arrays."""
    arrays = list(arrays)
    arrays[A_index] = A
    return tuple(arrays)

def _compute_diff(grads, grads_approx):
    """Computes the difference between the gradients and their approximation."""
    grad = _flatten(grads)
    grad_approx = _flatten(grads_approx)
    diff = np.linalg.norm(grad_approx - grad)
    diff_norm = np.linalg.norm(grad_approx) + np.linalg.norm(grad)
    diff /= diff_norm
    return diff

def _flatten(arrays):
    """Flattens the arrays into a single vector."""
    sizes = [np.prod(np.array(A.shape)) for A in arrays]
    m = sum(sizes)
    grad = np.empty(m)
    offset = 0
    for A, size in zip(arrays, sizes):
        end = offset + size
        grad[offset:end] = A.reshape(-1)
        offset = end
    return grad
