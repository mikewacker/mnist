import unittest
import nn_activations

import numpy as np

import gradient_checking

class NNActivationsTestCase(unittest.TestCase):

    def testReluActivation_Forward(self):
        activation = nn_activations.activation("relu")
        Z = np.array([[1.0, -2.0], [0.0, 0.25]])
        A = activation.forward(Z)
        A_expected = np.array([[1.0, 0.0], [0.0, 0.25]])
        self.assertTrue((A == A_expected).all())

    def testReluActivation_GradientCheck(self):
        self._testActivation_GradientCheck("relu")

    def testSigmoidActivation_GradientCheck(self):
        self._testActivation_GradientCheck("sigmoid")

    def testTanhActivation_GradientCheck(self):
        self._testActivation_GradientCheck("tanh")

    def _testActivation_GradientCheck(self, fn):
        self._activation = nn_activations.activation(fn)
        np.random.seed(195262)
        Z = np.random.normal(0, 1, (3, 3))
        diff, _, _ = gradient_checking.check(
            self._activationCost, self._activationGradients, Z)
        self.assertLess(diff, 1e-7)

    def _activationCost(self, Z):
        A = self._activation.forward(Z)
        cost = np.sum(A)
        return cost

    def _activationGradients(self, Z):
        A = self._activation.forward(Z)
        dA = np.ones(A.shape)
        dZ = self._activation.backward(dA)
        return (dZ,)

    def testActivationError_UnknownFunction(self):
        with self.assertRaises(ValueError):
            nn_activations.activation("dne")

    def testSigmoid(self):
        log_odds = np.log(np.array([[3.0, 4.0], [1.0, 0.25]]))
        A = nn_activations._sigmoid(log_odds)
        expected_prob = np.array([[0.75, 0.80], [0.50, 0.20]])
        self.assertArraysEqual(A, expected_prob)

    def testSoftmax(self):
        log_odds = np.log(np.array([[1.0, 3.0, 1.0], [2.0, 5.0, 3.0]]))
        A = nn_activations._softmax(log_odds)
        expected_prob = np.array([[0.2, 0.6, 0.2], [0.2, 0.5, 0.3]])
        self.assertArraysEqual(A, expected_prob)

    def assertArraysEqual(self, A, expected_A):
        A_diff = np.abs(A - expected_A)
        self.assertLess(np.max(A_diff), 1e-8)
