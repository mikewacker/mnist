import unittest
import numpy.testing as npt
from .testing import gradient_checking
from . import nn_layers

import numpy as np

class NNLayersTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(848131)

    ####
    # Layers
    ####

    def testDenseLayer(self):
        layer = nn_layers.dense(10, 4)
        self.assertEqual(layer.shape_in, (10,))
        self.assertEqual(layer.shape_out, (4,))
        self.assertEqual(layer.num_params, 44)
        self._testActivatedLayer_GradientCheck(layer)

    def testConvolution2DLayer(self):
        layer = nn_layers.convolution_2d(5, 5, 3, 6, kernel_size=3, padding=1)
        self.assertEqual(layer.shape_in, (5, 5, 3))
        self.assertEqual(layer.shape_out, (5, 5, 6))
        self.assertEqual(layer.num_params, 168)
        self._testActivatedLayer_GradientCheck(layer)

    def testMaxPool2DLayer(self):
        layer = nn_layers.max_pool_2d(4, 4, 3, pool_size=2)
        self.assertEqual(layer.shape_in, (4, 4, 3))
        self.assertEqual(layer.shape_out, (2, 2, 3))
        self.assertEqual(layer.num_params, 0)
        self._testActivatedLayer_GradientCheck(layer)

    def testFlattenLayer(self):
        layer = nn_layers.flatten((2, 2, 4))
        self.assertEqual(layer.shape_in, (2, 2, 4))
        self.assertEqual(layer.shape_out, (16,))
        self.assertEqual(layer.num_params, 0)
        self._testActivatedLayer_GradientCheck(layer)

    def testBinaryOutputLayer(self):
        layer = nn_layers.binary_output(10)
        self.assertEqual(layer.shape_in, (10,))
        self.assertEqual(layer.shape_out, (1,))
        self.assertEqual(layer.num_params, 11)
        self._testOutputLayer_GradientCheck(layer)

    def testMulticlassOutputLayer(self):
        layer = nn_layers.multiclass_output(10, 4)
        self.assertEqual(layer.shape_in, (10,))
        self.assertEqual(layer.shape_out, (4,))
        self.assertEqual(layer.num_params, 44)
        self._testOutputLayer_GradientCheck(layer)

    def _testActivatedLayer_GradientCheck(self, layer):
        A_prev = self._createAPrev(layer)
        self._layer = layer
        diff, _, _ = gradient_checking.check(
            self._activatedLayerCost, self._activatedLayerGradients,
            A_prev, *layer.weights)
        self.assertLess(diff, 1e-7)

    def _testOutputLayer_GradientCheck(self, layer):
        layer.Y = self._createY(layer)
        A_prev = self._createAPrev(layer)
        self._layer = layer
        diff, _, _ = gradient_checking.check(
            self._outputLayerCost, self._outputLayerGradients,
            A_prev, *layer.weights)
        self.assertLess(diff, 1e-7)

    def _createY(self, layer):
        Y_shape = (3, layer.C)
        Y = np.zeros(Y_shape)
        Y[range(3), np.random.randint(layer.C, size=3)] = 1.
        if not layer.is_multiclass:
            Y = Y[:, 0:1]
        return Y

    def _createAPrev(self, layer):
        A_prev_shape = (3, *layer.shape_in)
        A_prev = np.random.normal(0, 1, A_prev_shape)
        return A_prev

    def _activatedLayerCost(self, A_prev, *weights):
        layer = self._layer
        layer.weights = weights
        A = self._layer.forward(A_prev)
        cost = np.sum(A)
        return cost

    def _activatedLayerGradients(self, A_prev, *weights):
        layer = self._layer
        layer.weights = weights
        A = layer.forward(A_prev)
        dA = np.ones(A.shape)
        dA_prev, grads = layer.backward(dA, True)
        return (dA_prev, *grads)

    def _outputLayerCost(self, A_prev, *weights):
        layer = self._layer
        layer.weights = weights
        layer.forward(A_prev)
        return layer.cost

    def _outputLayerGradients(self, A_prev, *weights):
        layer = self._layer
        layer.weights = weights
        layer.forward(A_prev)
        dA_prev, grads = layer.backward(None, True)
        return (dA_prev, *grads)

    ####
    # Layer shapes
    ####

    def testCheckLayerShapes_OK(self):
        layers = [
            nn_layers.convolution_2d(10, 10, 3, 6, kernel_size=3, padding=1),
            nn_layers.max_pool_2d(10, 10, 6, pool_size=2),
            nn_layers.flatten((5, 5, 6)),
            nn_layers.dense(150, 50),
            nn_layers.binary_output(50),
        ]
        nn_layers.check_layer_shapes(layers)

    def testCheckLayerShapes_Misaligned(self):
        layers = [
            nn_layers.convolution_2d(10, 10, 3, 6, kernel_size=3, padding=1),
            nn_layers.max_pool_2d(10, 10, 6, pool_size=2),
            nn_layers.dense(150, 50),
        ]
        with self.assertRaises(ValueError):
            nn_layers.check_layer_shapes(layers)

    def testCheckInputShape_OK(self):
        layer = nn_layers.dense(10, 5)
        X = np.zeros((3, 10))
        nn_layers.check_input_shape(X, layer)

    def testCheckInputShape_Misaligned(self):
        layer = nn_layers.dense(10, 5)
        X = np.zeros((3, 9))
        with self.assertRaises(ValueError):
            nn_layers.check_input_shape(X, layer)

    def testOnehotOutput_Binary(self):
        output = nn_layers.binary_output(10)
        y = np.array([0, 1, 0])
        Y = nn_layers.onehot_output(y, output)
        expected_Y = np.array([[0], [1], [0]])
        npt.assert_equal(Y, expected_Y)

    def testOnehotOutput_Multiclass(self):
        output = nn_layers.multiclass_output(10, 3)
        y = np.array([1, 0, 2])
        Y = nn_layers.onehot_output(y, output)
        expected_Y = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
        npt.assert_equal(Y, expected_Y)

    ####
    # Layer composition
    ####

    def testActivatedLayer_Persistence(self):
        layer = nn_layers.dense(10, 4)
        nn_dict = {}
        layer.save_weights(nn_dict, 0)
        layer.load_weights(nn_dict, 0)

    def testOutputLayer_Predict(self):
        layer = nn_layers.multiclass_output(10, 4)
        A_prev = self._createAPrev(layer)
        layer.forward(A_prev)
        self.assertEqual(layer.pred.shape, (3,))
        self.assertEqual(layer.prob.shape, (3, 4))
