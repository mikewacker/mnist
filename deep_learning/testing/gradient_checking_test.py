import unittest
import numpy.testing as npt
from . import gradient_checking

import numpy as np

class GradientCheckingTestCase(unittest.TestCase):
    """Unit tests for gradient checking."""

    def testCheckGradients(self):
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        X = np.array([[5.0], [6.0]])
        diff, _, _ = gradient_checking.check(self._cost, self._grads, W, X)
        self.assertLess(diff, 1e-7)

    def _cost(self, W, X):
        """Sample cost function."""
        Y = W @ X
        cost = np.sum(Y)
        return cost

    def _grads(self, W, X):
        """Sample gradients function."""
        # If the cost is the sum of Y, then dY is an array of 1s.
        dY = np.ones((2, 1))
        dW = dY @ X.T
        dX = W.T @ dY
        return (dW, dX)

    ###
    # Subunit tests
    ###

    def testApproximateGradients(self):
        arrays = (np.array([[1.0, 2.0]]), np.array([[3.0], [4.0]]))
        cost_fn = lambda *arrays: sum(np.prod(A) for A in arrays)
        grads = gradient_checking._approximate_gradients(cost_fn, arrays, 0.1)
        expected_grads = (np.array([[2.0, 1.0]]), np.array([[4.0], [3.0]]))
        for dA, expected_dA in zip(grads, expected_grads):
            npt.assert_almost_equal(dA, expected_dA)

    def testComputeDiff(self):
        arrays1 = (np.array([[0.0, 1.0]]), np.array([[2.0], [3.0]]))
        arrays2 = (np.array([[1.0, 1.0]]), np.array([[1.0], [1.0]]))
        diff = gradient_checking._compute_diff_arrays(arrays1, arrays2)
        expected_diff = np.sqrt(6) / (np.sqrt(14) + 2)
        npt.assert_almost_equal(diff, expected_diff)
