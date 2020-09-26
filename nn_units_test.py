import unittest
import numpy.testing as npt
import nn_units

import numpy as np

import gradient_checking

class NNUnitsTestCase(unittest.TestCase):

    def testBaseUnit_Properties(self):
        W = np.zeros((3, 3, 3, 6))
        b = np.zeros((1, 1, 1, 6))
        unit = nn_units._BaseUnit((5, 5, 3), (3, 3, 6), (W, b))

        self.assertEqual(unit.shape_in, (5, 5, 3))
        self.assertEqual(unit.shape_out, (3, 3, 6))
        self.assertEqual(unit.num_params, 168)
        self.assertEqual(len(unit.weights), 2)
        self.assertEqual(unit.weights[0].shape, (3, 3, 3, 6))
        self.assertEqual(unit.weights[1].shape, (1, 1, 1, 6))

    def testBaseUnit_Properties_NoWeights(self):
        unit = nn_units._BaseUnit((3, 3, 6), 54)

        self.assertEqual(unit.shape_in, (3, 3, 6))
        self.assertEqual(unit.shape_out, (54,))
        self.assertEqual(unit.num_params, 0)
        self.assertEqual(len(unit.weights), 0)

    def testBaseUnit_SetWeights(self):
        W_0 = np.zeros((4, 2))
        b_0 = np.zeros((1, 2))
        W = np.ones((4, 2))
        b = np.ones((1, 2))
        unit = nn_units._BaseUnit(4, 2, (W_0, b_0))
        unit.weights = (W, b)

        self.assertEqual(unit.weights[0][0, 0], 1.)

    def testBaseUnit_SetWeights_NoWeights(self):
        unit = nn_units._BaseUnit(4, 2)
        unit.weights = ()

    def testBaseUnit_Persistence(self):
        W_0 = np.zeros((4, 2))
        b_0 = np.zeros((1, 2))
        W = np.ones((4, 2))
        b = np.ones((1, 2))
        nn_dict = {}
        unit1 = nn_units._BaseUnit(4, 2, (W, b))
        unit2 = nn_units._BaseUnit(4, 2, (W_0, b_0))
        unit1.save_weights(nn_dict, 0)
        unit2.load_weights(nn_dict, 0)

        self.assertEqual(unit2.weights[0][0, 0], 1.)
        expected_keys = {"layer 1: weights[0]", "layer 1: weights[1]"}
        self.assertCountEqual(nn_dict.keys(), expected_keys)

    def testBaseUnit_Persistence_NoWeights(self):
        nn_dict = {}
        unit1 = nn_units._BaseUnit(4, 2)
        unit2 = nn_units._BaseUnit(4, 2)
        unit1.save_weights(nn_dict, 0)
        unit2.load_weights(nn_dict, 0)

        self.assertFalse(nn_dict)

    def testBaseUnitError_InvalidWeights_Create(self):
        W = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            nn_units._BaseUnit(4, 2, W)

    def testBaseUnitError_InvalidWeights_SetWeights(self):
        W1 = np.zeros((2, 2))
        W2 = np.zeros((1, 2))
        unit = nn_units._BaseUnit(4, 2, (W1,))
        with self.assertRaises(ValueError):
            unit.weights = W1
        with self.assertRaises(ValueError):
            unit.weights = (W1, W1)
        with self.assertRaises(ValueError):
            unit.weights = (W2,)

    def testBaseUnitError_InvalidWeights_LoadWeights(self):
        W1 = np.zeros((2, 2))
        W2 = np.zeros((1, 2))
        unit = nn_units._BaseUnit(4, 2, (W1,))
        with self.assertRaises(ValueError):
            unit.load_weights({}, 0)
        with self.assertRaises(ValueError):
            unit.load_weights({"layer 1: weights[0]": W2}, 0)

    def testDenseUnit_Shape(self):
        unit = nn_units.dense(4, 3)
        self.assertEqual(unit.shape_in, (4,))
        self.assertEqual(unit.shape_out, (3,))

    def testDenseUnit_Forward(self):
        unit = nn_units.dense(3, 2)
        A_prev = np.array([[1., 2., 3.], [3., 2., 1.]])
        W = np.array([[1., 3.], [2., 2.], [3., 1.]])
        b = np.array([[1., 2.]])
        unit.weights = (W, b)
        Z = unit.forward(A_prev)
        expected_Z = np.array([[15., 12.], [11., 16.]])
        npt.assert_almost_equal(Z, expected_Z)

    def testDenseUnit_UnchainedBackward(self):
        unit = nn_units.dense(4, 2)
        self._testUnit_UnchainedBackward(unit)

    def testDenseUnit_GradientCheck(self):
        unit = nn_units.dense(4, 2)
        self._testUnit_GradientCheck(unit)

    def testFlattenUnit_Shape(self):
        unit = nn_units.flatten((4, 4, 2))
        self.assertEqual(unit.shape_in, (4, 4, 2))
        self.assertEqual(unit.shape_out, (32,))

    def testFlattenUnit_Forward(self):
        unit = nn_units.flatten((4, 4, 2))
        A_prev = np.zeros((3, 4, 4, 2))
        Z = unit.forward(A_prev)
        self.assertEqual(Z.shape, (3, 32))

    def testFlattenUnit_UnchainedBackward(self):
        unit = nn_units.flatten((4, 4, 2))
        self._testUnit_UnchainedBackward(unit)

    def testFlattenUnit_GradientCheck(self):
        unit = nn_units.flatten((4, 4, 2))
        self._testUnit_GradientCheck(unit)

    def testMaxPool2DUnit_Shape(self):
        unit = nn_units.max_pool_2d(28, 28, 6, pool_size=2)
        self.assertEqual(unit.shape_in, (28, 28, 6))
        self.assertEqual(unit.shape_out, (14, 14, 6))

    def testMaxPool2DUnit_Shape_DifferentStride(self):
        unit = nn_units.max_pool_2d(28, 28, 6, pool_size=5, stride=3)
        self.assertEqual(unit.shape_in, (28, 28, 6))
        self.assertEqual(unit.shape_out, (8, 8, 6))

    def testMaxPool2DUnit_Forward(self):
        unit = nn_units.max_pool_2d(4, 4, 1, pool_size=2)
        A_prev = np.array([
            [
                [[1.], [2.], [5.], [2.]],
                [[3.], [4.], [3.], [4.]],
                [[5.], [6.], [5.], [6.]],
                [[3.], [4.], [7.], [4.]],
            ],
            [
                [[5.], [6.], [5.], [6.]],
                [[3.], [4.], [7.], [4.]],
                [[1.], [2.], [5.], [2.]],
                [[3.], [4.], [3.], [4.]],
            ],
        ])
        Z = unit.forward(A_prev)
        expected_Z = np.array([
            [
                [[4.], [5.]],
                [[6.], [7.]],
            ],
            [
                [[6.], [7.]],
                [[4.], [5.]],
            ],
        ])
        npt.assert_almost_equal(Z, expected_Z)

    def testMaxPool2DUnit_UnchainedBackward(self):
        unit = nn_units.max_pool_2d(4, 4, 2, pool_size=2)
        self._testUnit_UnchainedBackward(unit)

    def testMaxPool2DUnit_GradientCheck(self):
        unit = nn_units.max_pool_2d(4, 4, 2, pool_size=2)
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
        np.random.seed(460072)
        A_prev_shape = (3, *unit.shape_in)
        A_prev = np.random.normal(0, 1, A_prev_shape)
        return A_prev

    def _unitCost(self, A_prev, *weights):
        self._unit.weights = weights
        Z = self._unit.forward(A_prev)
        cost = np.sum(Z)
        return cost

    def _unitGradients(self, A_prev, *weights):
        self._unit.weights = weights
        Z = self._unit.forward(A_prev)
        dZ = np.ones(Z.shape)
        dA_prev, grads = self._unit.backward(dZ, True)
        return (dA_prev, *grads)

    def testGlorotNormalInitialization(self):
        np.random.seed(692616)
        W = nn_units._glorot_normal_initialization((500, 300), 500, 300)
        self.assertEqual(W.shape, (500, 300))
        self.assertAlmostEqual(np.mean(W), 0, places=3)
        self.assertAlmostEqual(np.std(W), 0.05, places=3)
