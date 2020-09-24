import unittest
import nn_units

import numpy as np

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

        self.assertEqual(unit.weights[0][0, 0], 1.0)

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

        self.assertEqual(unit2.weights[0][0, 0], 1.0)
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
