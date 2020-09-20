import unittest
import nn_optimizers

import numpy as np

class NNOptimizerTestCase(unittest.TestCase):

    def testGradientDescentStep(self):
        stepper = nn_optimizers._GradientDescent((2, 2))

        dW, _, _ = self._createGradients()
        step = stepper.step(dW)

        self.assertArraysEqual(step, dW)

    def testGradientDescentEmptyState(self):
        stepper = nn_optimizers._GradientDescent((2, 2))
        self.assertEqual(stepper.state, ())
        stepper.state = ()

    def testAdamStep(self):
        stepper = nn_optimizers._Adam((2, 2), 0.9, 0.999, 1e-8)

        dW1, dW2, dW3 = self._createGradients()
        stepper.step(dW1)
        stepper.step(dW2)
        step = stepper.step(dW3)

        expected_step = self._getExpectedAdamStep(dW1, dW2, dW3)
        self.assertArraysEqual(step, expected_step)

    def testAdamStep_Persistence(self):
        stepper1 = nn_optimizers._Adam((2, 2), 0.9, 0.999, 1e-8)
        stepper2 = nn_optimizers._Adam((2, 2), 0.9, 0.999, 1e-8)

        dW1, dW2, dW3 = self._createGradients()
        stepper1.step(dW1)
        stepper1.step(dW2)
        state = stepper1.state
        stepper2.state = state
        step = stepper2.step(dW3)

        expected_step = self._getExpectedAdamStep(dW1, dW2, dW3)
        self.assertArraysEqual(step, expected_step)

    def _createGradients(self):
        dW1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        dW2 = np.array([[4.0, 1.0], [2.0, 3.0]])
        dW3 = np.array([[3.0, 4.0], [1.0, 2.0]])
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

    def assertArraysEqual(self, A, expected_A):
        A_diff = np.abs(A - expected_A)
        self.assertLess(np.max(A_diff), 1e-8)
