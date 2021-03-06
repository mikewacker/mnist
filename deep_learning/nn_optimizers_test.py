import unittest
import numpy.testing as npt
from . import nn_optimizers

import numpy as np

class NNOptimizersTestCase(unittest.TestCase):

    ####
    # Optimizers
    ####

    def testOptimizer_GradientDescent(self):
        optimizer = nn_optimizers.gradient_descent()
        layer = MockLayer()
        optimizer.init_steppers([layer])

        dW, _, _ = self._createGradients()
        W_prev = layer.weights[0]
        optimizer.update_weights(0, layer, (dW,), 0.01, 0)

        expected_W = W_prev - 0.01 * dW
        npt.assert_almost_equal(layer.weights[0], expected_W)

    def testOptimizer_Adam(self):
        optimizer = nn_optimizers.adam()
        layer = MockLayer()
        optimizer.init_steppers([layer])

        dW1, dW2, dW3 = self._createGradients()
        optimizer.update_weights(0, layer, (dW1,), 0.01, 0)
        optimizer.update_weights(0, layer, (dW2,), 0.01, 0)
        W_prev = layer.weights[0]
        optimizer.update_weights(0, layer, (dW3,), 0.01, 0)

        expected_step = self._getExpectedAdamStep(dW1, dW2, dW3)
        expected_W = W_prev - 0.01 * expected_step
        npt.assert_almost_equal(layer.weights[0], expected_W)

    def testOptimizer_WeightDecay(self):
        optimizer = nn_optimizers.gradient_descent()
        layer = MockLayer()
        optimizer.init_steppers([layer])

        dW, _, _ = self._createGradients()
        W_prev = layer.weights[0]
        optimizer.update_weights(0, layer, (dW,), 0.01, 0.0001)

        expected_W = (1 - 0.01 * 0.0001) * W_prev - 0.01 * dW
        npt.assert_almost_equal(layer.weights[0], expected_W)

    def testOptimizer_WeightDecay_Unregularized(self):
        optimizer = nn_optimizers.gradient_descent()
        layer = MockLayer(False)
        optimizer.init_steppers([layer])

        dW, _, _ = self._createGradients()
        W_prev = layer.weights[0]
        optimizer.update_weights(0, layer, (dW,), 0.01, 0.0001)

        expected_W = W_prev - 0.01 * dW
        npt.assert_almost_equal(layer.weights[0], expected_W)

    def testOptimizer_Persistence(self):
        optimizer1 = nn_optimizers.adam()
        optimizer2 = nn_optimizers.adam()
        layer = MockLayer()
        optimizer1.init_steppers([layer])
        optimizer2.init_steppers([layer])
        nn_dict = {}

        dW1, dW2, dW3 = self._createGradients()
        optimizer1.update_weights(0, layer, (dW1,), 0.01, 0)
        optimizer1.update_weights(0, layer, (dW2,), 0.01, 0)
        optimizer1.save_state(nn_dict)
        optimizer2.load_state(nn_dict)
        W_prev = layer.weights[0]
        optimizer2.update_weights(0, layer, (dW3,), 0.01, 0)

        expected_step = self._getExpectedAdamStep(dW1, dW2, dW3)
        expected_W = W_prev - 0.01 * expected_step
        npt.assert_almost_equal(layer.weights[0], expected_W)
        expected_keys = [
            "layer 1: optimizer-adam[0][0]",
            "layer 1: optimizer-adam[0][1]",
            "layer 1: optimizer-adam[0][2]",
        ]
        self.assertCountEqual(nn_dict.keys(), expected_keys)

    def testOptimizer_Persistence_Stateless(self):
        optimizer1 = nn_optimizers.gradient_descent()
        optimizer2 = nn_optimizers.gradient_descent()
        layer = MockLayer()
        optimizer1.init_steppers([layer])
        optimizer2.init_steppers([layer])
        nn_dict = {}

        optimizer1.save_state(nn_dict)
        optimizer2.load_state(nn_dict)

        self.assertFalse(nn_dict)

    ####
    # Steppers
    ####

    def testGradientDescent_Step(self):
        stepper = nn_optimizers._GradientDescent((2, 2))

        dW, _, _ = self._createGradients()
        step = stepper.step(dW)

        npt.assert_almost_equal(step, dW)

    def testGradientDescent_EmptyState(self):
        stepper = nn_optimizers._GradientDescent((2, 2))
        self.assertEqual(stepper.state, ())
        stepper.state = ()

    def testAdam_Step(self):
        stepper = nn_optimizers._Adam((2, 2), 0.9, 0.999, 1e-8)

        dW1, dW2, dW3 = self._createGradients()
        stepper.step(dW1)
        stepper.step(dW2)
        step = stepper.step(dW3)

        expected_step = self._getExpectedAdamStep(dW1, dW2, dW3)
        npt.assert_almost_equal(step, expected_step)

    def testAdam_Step_Persistence(self):
        stepper1 = nn_optimizers._Adam((2, 2), 0.9, 0.999, 1e-8)
        stepper2 = nn_optimizers._Adam((2, 2), 0.9, 0.999, 1e-8)

        dW1, dW2, dW3 = self._createGradients()
        stepper1.step(dW1)
        stepper1.step(dW2)
        state = stepper1.state
        stepper2.state = state
        step = stepper2.step(dW3)

        expected_step = self._getExpectedAdamStep(dW1, dW2, dW3)
        npt.assert_almost_equal(step, expected_step)

    def _createGradients(self):
        dW1 = np.array([[1., 2.], [3., 4.]])
        dW2 = np.array([[4., 1.], [2., 3.]])
        dW3 = np.array([[3., 4.], [1., 2.]])
        return dW1, dW2, dW3

    def _getExpectedAdamStep(self, dW1, dW2, dW3):
        vdW = sum([
            0.1 * dW3,
            0.1 * 0.9 * dW2,
            0.1 * 0.9 ** 2 * dW1])
        vdW_bc = vdW / (1 - 0.9 ** 3)
        sdW = sum([
            0.001 * np.square(dW3),
            0.001 * 0.999 * np.square(dW2),
            0.001 * 0.999 ** 2 * np.square(dW1)])
        sdW_bc = sdW / (1 - 0.999 ** 3)
        step = vdW_bc / (np.sqrt(sdW_bc) + 1e-8)
        return step

class MockLayer(object):

    def __init__(self, regularize=True):
        W = np.array([[2., 3.], [4., 1.]])
        self.weights = (W,)
        self.regularized = (regularize,)
