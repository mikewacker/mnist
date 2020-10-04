import unittest
import numpy.testing as npt
from .testing import gradient_checking
from . import _nn_units

import numpy as np

class NNUnitsTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(460072)

    ####
    # Base unit
    ####

    def testBaseUnit_Properties(self):
        W = np.zeros((3, 3, 3, 6))
        b = np.zeros((1, 1, 1, 6))
        unit = _nn_units._BaseUnit((5, 5, 3), (3, 3, 6), (W, b))

        self.assertEqual(unit.shape_in, (5, 5, 3))
        self.assertEqual(unit.shape_out, (3, 3, 6))
        self.assertEqual(unit.num_params, 168)
        self.assertEqual(len(unit.weights), 2)
        self.assertEqual(unit.weights[0].shape, (3, 3, 3, 6))
        self.assertEqual(unit.weights[1].shape, (1, 1, 1, 6))

    def testBaseUnit_Properties_NoWeights(self):
        unit = _nn_units._BaseUnit((3, 3, 6), 54)

        self.assertEqual(unit.shape_in, (3, 3, 6))
        self.assertEqual(unit.shape_out, (54,))
        self.assertEqual(unit.num_params, 0)
        self.assertEqual(len(unit.weights), 0)

    def testBaseUnit_SetWeights(self):
        W_0 = np.zeros((4, 2))
        b_0 = np.zeros((1, 2))
        W = np.ones((4, 2))
        b = np.ones((1, 2))
        unit = _nn_units._BaseUnit(4, 2, (W_0, b_0))
        unit.weights = (W, b)

        self.assertEqual(unit.weights[0][0, 0], 1.)

    def testBaseUnit_SetWeights_NoWeights(self):
        unit = _nn_units._BaseUnit(4, 2)
        unit.weights = ()

    def testBaseUnit_Persistence(self):
        W_0 = np.zeros((4, 2))
        b_0 = np.zeros((1, 2))
        W = np.ones((4, 2))
        b = np.ones((1, 2))
        nn_dict = {}
        unit1 = _nn_units._BaseUnit(4, 2, (W, b))
        unit2 = _nn_units._BaseUnit(4, 2, (W_0, b_0))
        unit1.save_weights(nn_dict, 0)
        unit2.load_weights(nn_dict, 0)

        self.assertEqual(unit2.weights[0][0, 0], 1.)
        expected_keys = {"layer 1: weights[0]", "layer 1: weights[1]"}
        self.assertCountEqual(nn_dict.keys(), expected_keys)

    def testBaseUnit_Persistence_NoWeights(self):
        nn_dict = {}
        unit1 = _nn_units._BaseUnit(4, 2)
        unit2 = _nn_units._BaseUnit(4, 2)
        unit1.save_weights(nn_dict, 0)
        unit2.load_weights(nn_dict, 0)

        self.assertFalse(nn_dict)

    def testBaseUnitError_InvalidWeights_Create(self):
        W = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            _nn_units._BaseUnit(4, 2, W)

    def testBaseUnitError_InvalidWeights_SetWeights(self):
        W1 = np.zeros((2, 2))
        W2 = np.zeros((1, 2))
        unit = _nn_units._BaseUnit(4, 2, (W1,))
        with self.assertRaises(ValueError):
            unit.weights = W1
        with self.assertRaises(ValueError):
            unit.weights = (W1, W1)
        with self.assertRaises(ValueError):
            unit.weights = (W2,)

    def testBaseUnitError_InvalidWeights_LoadWeights(self):
        W1 = np.zeros((2, 2))
        W2 = np.zeros((1, 2))
        unit = _nn_units._BaseUnit(4, 2, (W1,))
        with self.assertRaises(ValueError):
            unit.load_weights({}, 0)
        with self.assertRaises(ValueError):
            unit.load_weights({"layer 1: weights[0]": W2}, 0)

    ####
    # Units
    ####

    def testDenseUnit_Shape(self):
        unit = _nn_units.dense(4, 3)
        self.assertEqual(unit.shape_in, (4,))
        self.assertEqual(unit.shape_out, (3,))

    def testDenseUnit_Forward(self):
        unit = _nn_units.dense(3, 2)
        A_prev = np.array([[1., 2., 3.], [3., 2., 1.]])
        W = np.array([[1., 3.], [2., 2.], [3., 1.]])
        b = np.array([[1., 2.]])
        unit.weights = (W, b)
        Z = unit.forward(A_prev)
        expected_Z = np.array([[15., 12.], [11., 16.]])
        npt.assert_almost_equal(Z, expected_Z)

    def testDenseUnit_UnchainedBackward(self):
        unit = _nn_units.dense(4, 2)
        self._testUnit_UnchainedBackward(unit)

    def testDenseUnit_GradientCheck(self):
        unit = _nn_units.dense(4, 2)
        self._testUnit_GradientCheck(unit)

    def testConvolution2DUnit_Shape(self):
        unit = _nn_units.convolution_2d(
            7, 7, 2, 5, kernel_size=3, stride=2, padding=1)
        self.assertEqual(unit.shape_in, (7, 7, 2))
        self.assertEqual(unit.shape_out, (4, 4, 5))

    def testConvolution2DUnit_Forward(self):
        unit = _nn_units.convolution_2d(3, 3, 2, 2, kernel_size=2)
        A_prev = np.random.normal(0, 1, (2, 3, 3, 2))
        W, b = unit.weights
        Z = unit.forward(A_prev)
        expected_Z = np.array([
            np.sum(W[:, :, :, 0] * A_prev[0, 0:2, 0:2, :]) + b[0],
            np.sum(W[:, :, :, 1] * A_prev[0, 0:2, 0:2, :]) + b[1],
            np.sum(W[:, :, :, 0] * A_prev[0, 0:2, 1:3, :]) + b[0],
            np.sum(W[:, :, :, 1] * A_prev[0, 0:2, 1:3, :]) + b[1],
            np.sum(W[:, :, :, 0] * A_prev[0, 1:3, 0:2, :]) + b[0],
            np.sum(W[:, :, :, 1] * A_prev[0, 1:3, 0:2, :]) + b[1],
            np.sum(W[:, :, :, 0] * A_prev[0, 1:3, 1:3, :]) + b[0],
            np.sum(W[:, :, :, 1] * A_prev[0, 1:3, 1:3, :]) + b[1],
            np.sum(W[:, :, :, 0] * A_prev[1, 0:2, 0:2, :]) + b[0],
            np.sum(W[:, :, :, 1] * A_prev[1, 0:2, 0:2, :]) + b[1],
            np.sum(W[:, :, :, 0] * A_prev[1, 0:2, 1:3, :]) + b[0],
            np.sum(W[:, :, :, 1] * A_prev[1, 0:2, 1:3, :]) + b[1],
            np.sum(W[:, :, :, 0] * A_prev[1, 1:3, 0:2, :]) + b[0],
            np.sum(W[:, :, :, 1] * A_prev[1, 1:3, 0:2, :]) + b[1],
            np.sum(W[:, :, :, 0] * A_prev[1, 1:3, 1:3, :]) + b[0],
            np.sum(W[:, :, :, 1] * A_prev[1, 1:3, 1:3, :]) + b[1],
        ]).reshape((2, 2, 2, 2))
        npt.assert_almost_equal(Z, expected_Z)

    def testConvolution2DUnit_UnchainedBackward(self):
        unit = _nn_units.convolution_2d(
            7, 7, 2, 5, kernel_size=3, stride=2, padding=1)
        self._testUnit_UnchainedBackward(unit)

    def testConvolution2DUnit_GradientCheck(self):
        unit = _nn_units.convolution_2d(
            7, 7, 2, 5, kernel_size=3, stride=2, padding=1)
        self._testUnit_GradientCheck(unit)

    def testMaxPool2DUnit_Shape(self):
        unit = _nn_units.max_pool_2d(28, 28, 6, pool_size=2)
        self.assertEqual(unit.shape_in, (28, 28, 6))
        self.assertEqual(unit.shape_out, (14, 14, 6))

    def testMaxPool2DUnit_Shape_DifferentStride(self):
        unit = _nn_units.max_pool_2d(28, 28, 6, pool_size=5, stride=3)
        self.assertEqual(unit.shape_in, (28, 28, 6))
        self.assertEqual(unit.shape_out, (8, 8, 6))

    def testMaxPool2DUnit_Forward(self):
        unit = _nn_units.max_pool_2d(4, 4, 2, pool_size=2)
        A_prev = np.random.normal(0, 1, (2, 4, 4, 2))
        Z = unit.forward(A_prev)
        expected_Z = np.array([
            np.max(A_prev[0, 0:2, 0:2, 0]), np.max(A_prev[0, 0:2, 0:2, 1]),
            np.max(A_prev[0, 0:2, 2:4, 0]), np.max(A_prev[0, 0:2, 2:4, 1]),
            np.max(A_prev[0, 2:4, 0:2, 0]), np.max(A_prev[0, 2:4, 0:2, 1]),
            np.max(A_prev[0, 2:4, 2:4, 0]), np.max(A_prev[0, 2:4, 2:4, 1]),
            np.max(A_prev[1, 0:2, 0:2, 0]), np.max(A_prev[1, 0:2, 0:2, 1]),
            np.max(A_prev[1, 0:2, 2:4, 0]), np.max(A_prev[1, 0:2, 2:4, 1]),
            np.max(A_prev[1, 2:4, 0:2, 0]), np.max(A_prev[1, 2:4, 0:2, 1]),
            np.max(A_prev[1, 2:4, 2:4, 0]), np.max(A_prev[1, 2:4, 2:4, 1]),
        ]).reshape((2, 2, 2, 2))
        npt.assert_almost_equal(Z, expected_Z)

    def testMaxPool2DUnit_UnchainedBackward(self):
        unit = _nn_units.max_pool_2d(4, 4, 2, pool_size=2)
        self._testUnit_UnchainedBackward(unit)

    def testMaxPool2DUnit_GradientCheck(self):
        unit = _nn_units.max_pool_2d(4, 4, 2, pool_size=2)
        self._testUnit_GradientCheck(unit)

    def testFlattenUnit_Shape(self):
        unit = _nn_units.flatten((4, 4, 2))
        self.assertEqual(unit.shape_in, (4, 4, 2))
        self.assertEqual(unit.shape_out, (32,))

    def testFlattenUnit_Forward(self):
        unit = _nn_units.flatten((4, 4, 2))
        A_prev = np.zeros((3, 4, 4, 2))
        Z = unit.forward(A_prev)
        self.assertEqual(Z.shape, (3, 32))

    def testFlattenUnit_UnchainedBackward(self):
        unit = _nn_units.flatten((4, 4, 2))
        self._testUnit_UnchainedBackward(unit)

    def testFlattenUnit_GradientCheck(self):
        unit = _nn_units.flatten((4, 4, 2))
        self._testUnit_GradientCheck(unit)

    def _testUnit_UnchainedBackward(self, unit):
        A_prev = self._createAPrev(unit)
        Z = unit.forward(A_prev)
        dZ = np.ones(Z.shape)
        _, grads = unit.backward(dZ, True)
        dA_prev_unchained, grads_unchained = unit.backward(dZ, False)
        self.assertIsNone(dA_prev_unchained)
        for dW, dW_unchained in zip(grads, grads_unchained):
            npt.assert_equal(dW, dW_unchained)

    def _testUnit_GradientCheck(self, unit):
        A_prev = self._createAPrev(unit)
        self._unit = unit
        diff, _, _ = gradient_checking.check(
            self._unitCost, self._unitGradients, A_prev, *unit.weights)
        self.assertLess(diff, 1e-7)

    def _createAPrev(self, unit):
        A_prev_shape = (3, *unit.shape_in)
        A_prev = np.random.normal(0, 1, A_prev_shape)
        return A_prev

    def _unitCost(self, A_prev, *weights):
        unit = self._unit
        unit.weights = weights
        Z = unit.forward(A_prev)
        cost = np.sum(Z)
        return cost

    def _unitGradients(self, A_prev, *weights):
        unit = self._unit
        unit.weights = weights
        Z = unit.forward(A_prev)
        dZ = np.ones(Z.shape)
        dA_prev, grads = unit.backward(dZ, True)
        return (dA_prev, *grads)

    ####
    # Weight initialization
    ####

    def testGlorotUniformInitialization(self):
        W = _nn_units._glorot_uniform_initialization((500, 300), 500, 300)
        self.assertEqual(W.shape, (500, 300))
        self.assertAlmostEqual(np.mean(W), 0, places=3)
        self.assertAlmostEqual(np.std(W), 0.05, places=3)
