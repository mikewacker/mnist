import unittest
import numpy.testing as npt
import nn_activations

import numpy as np

import gradient_checking

class NNActivationsTestCase(unittest.TestCase):

    ####
    # Activations
    ####

    def testReluActivation_Forward(self):
        activation = nn_activations.activation("relu")
        Z = np.array([[1., -2.], [0., 0.25]])
        A = activation.forward(Z)
        expected_A = np.array([[1., 0.], [0., 0.25]])
        npt.assert_almost_equal(A, expected_A)

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
        Z = self._createZ(3)
        diff, _, _ = gradient_checking.check(
            self._activationCost, self._activationGradients, Z)
        self.assertLess(diff, 1e-7)

    def _createZ(self, num_cols):
        np.random.seed(195262)
        Z_shape = (3, num_cols)
        Z = np.random.normal(0, 1, Z_shape)
        return Z

    def _activationCost(self, Z):
        A = self._activation.forward(Z)
        cost = np.sum(A)
        return cost

    def _activationGradients(self, Z):
        A = self._activation.forward(Z)
        dA = np.ones(A.shape)
        dZ = self._activation.backward(dA)
        return (dZ,)

    ####
    # Outputs
    ####

    def testBinaryOutput_Properties(self):
        output = nn_activations.binary_output()
        self.assertFalse(output.is_multiclass)
        self.assertEqual(output.C, 2)

    def testBinaryOutput_Predict(self):
        output = nn_activations.binary_output()
        Z = np.log(np.array([[3.], [1.], [0.25]]))
        expected_pred = np.array([1, 0, 0])
        expected_prob = np.array([0.75, 0.5, 0.2])
        self._testOutput_Predict(output, Z, expected_pred, expected_prob)

    def testBinaryOutput_GradientCheck(self):
        output = nn_activations.binary_output()
        output.Y = np.array([[0.], [1.], [0.]])
        self._testOutput_GradientCheck(output)

    def testMulticlassOutput_Properties(self):
        output = nn_activations.multiclass_output(3)
        self.assertTrue(output.is_multiclass)
        self.assertEqual(output.C, 3)

    def testMulticlassOutput_Predict(self):
        output = nn_activations.multiclass_output(3)
        Z = np.log(np.array([[1., 2., 1.], [5., 3., 2.], [1., 1., 3.]]))
        expected_pred = np.array([1, 0, 2])
        expected_prob = np.array(
            [[0.25, 0.50, 0.25], [0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
        self._testOutput_Predict(output, Z, expected_pred, expected_prob)

    def testMulticlassOutput_GradientCheck(self):
        output = nn_activations.multiclass_output(3)
        output.Y = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self._testOutput_GradientCheck(output)

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
        npt.assert_almost_equal(pred, expected_pred)
        self.assertEqual(prob.shape, expected_prob.shape)
        npt.assert_almost_equal(prob, expected_prob)

    def _testOutput_GradientCheck(self, output):
        num_cols = output.C if output.is_multiclass else 1
        Z = self._createZ(num_cols)
        self._output = output
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

    ####
    # Activation functions
    ####

    def testSigmoid(self):
        log_odds = np.log(np.array([[3., 4.], [1., 0.25]]))
        A = nn_activations._sigmoid(log_odds)
        expected_prob = np.array([[0.75, 0.80], [0.50, 0.20]])
        npt.assert_almost_equal(A, expected_prob)

    def testSoftmax(self):
        log_odds = np.log(np.array([[1., 3., 1.], [2., 5., 3.]]))
        A = nn_activations._softmax(log_odds)
        expected_prob = np.array([[0.2, 0.6, 0.2], [0.2, 0.5, 0.3]])
        npt.assert_almost_equal(A, expected_prob)
