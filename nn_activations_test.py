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

    def testActivationError_UnknownFunction(self):
        with self.assertRaises(ValueError):
            nn_activations.activation("dne")

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

    def testBinaryOutput_Properties(self):
        output = nn_activations.binary_output()
        self.assertFalse(output.is_multiclass)
        self.assertEqual(output.C, 2)

    def testBinaryOutput_Predict(self):
        output = nn_activations.binary_output()
        Z = np.log(np.array([[3.0], [1.0], [0.25]]))
        expected_pred = np.array([1, 0, 0])
        expected_prob = np.array([0.75, 0.5, 0.2])
        self._testOutput_Predict(output, Z, expected_pred, expected_prob)

    def testBinaryOutput_GradientCheck(self):
        self._output = nn_activations.binary_output()
        self._output.Y = np.array([[0.0], [1.0], [0.0]])
        self._testOutput_GradientCheck(1)

    def testMulticlassOutput_Properties(self):
        output = nn_activations.multiclass_output(3)
        self.assertTrue(output.is_multiclass)
        self.assertEqual(output.C, 3)

    def testMulticlassOutput_Predict(self):
        output = nn_activations.multiclass_output(3)
        Z = np.log(np.array(
            [[1.0, 2.0, 1.0], [5.0, 3.0, 2.0], [1.0, 1.0, 3.0]]))
        expected_pred = np.array([1, 0, 2])
        expected_prob = np.array(
            [[0.25, 0.50, 0.25], [0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
        self._testOutput_Predict(output, Z, expected_pred, expected_prob)

    def testMulticlassOutput_GradientCheck(self):
        self._output = nn_activations.multiclass_output(3)
        self._output.Y = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self._testOutput_GradientCheck(3)

    def testMulticlasOutputError_IllegalC(self):
        with self.assertRaises(ValueError):
            nn_activations.multiclass_output(2)

    def testOutputError_YNotSet(self):
        output = nn_activations.binary_output()
        with self.assertRaises(RuntimeError):
            output.Y
        Z = np.zeros((3, 1))
        output.forward(Z)
        with self.assertRaises(RuntimeError):
            output.cost
        with self.assertRaises(RuntimeError):
            output.backward(None)

    def _testOutput_Predict(self, output, Z, expected_pred, expected_prob):
        output.forward(Z)
        pred = output.pred
        prob = output.prob
        self.assertEqual(pred.shape, expected_pred.shape)
        self.assertTrue((pred == expected_pred).all())
        self.assertEqual(prob.shape, expected_prob.shape)
        self.assertArraysEqual(prob, expected_prob)

    def _testOutput_GradientCheck(self, num_cols):
        np.random.seed(149773)
        Z = np.random.normal(0, 1, (3, num_cols))
        diff, _, _ = gradient_checking.check(
            self._outputCost, self._outputGradients, Z)
        self.assertLess(diff, 1e-7)

    def _outputCost(self, Z):
        self._output.forward(Z)
        cost = self._output.cost
        return cost

    def _outputGradients(self, Z):
        self._output.forward(Z)
        dZ = self._output.backward(None)
        return (dZ,)

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
