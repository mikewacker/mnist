import unittest
from . import gradient_checking

import numpy as np

class GradientCheckingTestCase(unittest.TestCase):

    def testCheckGradients(self):
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        X = np.array([[5.0], [6.0]])
        diff, _, _ = gradient_checking.check(self._cost, self._grads, W, X)
        self.assertLess(diff, 1e-7)

    def _cost(self, W, X):
        Y = W @ X
        cost = np.sum(Y)
        return cost

    def _grads(self, W, X):
        dY = np.ones((2, 1))
        dW = dY @ X.T
        dX = W.T @ dY
        return (dW, dX)
