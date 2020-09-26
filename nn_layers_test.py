import unittest
import numpy.testing as npt
import nn_layers

import numpy as np

import gradient_checking

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
        self._testActivatedLayer_GradientCheck(layer)

    def testConvolution2DLayer(self):
        layer = nn_layers.convolution_2d(5, 5, 3, 6, kernel_size=3, padding=1)
        self.assertEqual(layer.shape_in, (5, 5, 3))
        self.assertEqual(layer.shape_out, (5, 5, 6))
        self._testActivatedLayer_GradientCheck(layer)

    def testMaxPool2DLayer(self):
        layer = nn_layers.max_pool_2d(4, 4, 3, pool_size=2)
        self.assertEqual(layer.shape_in, (4, 4, 3))
        self.assertEqual(layer.shape_out, (2, 2, 3))
        self._testActivatedLayer_GradientCheck(layer)

    def testFlattenLayer(self):
        layer = nn_layers.flatten((2, 2, 4))
        self.assertEqual(layer.shape_in, (2, 2, 4))
        self.assertEqual(layer.shape_out, (16,))
        self._testActivatedLayer_GradientCheck(layer)

    def testBinaryOutpuLayer(self):
        layer = nn_layers.binary_output(10)
        self.assertEqual(layer.shape_in, (10,))
        self.assertEqual(layer.shape_out, (1,))
        self._testOutputLayer_GradientCheck(layer)

    def testMulticlassOutputLayer(self):
        layer = nn_layers.multiclass_output(10, 4)
        self.assertEqual(layer.shape_in, (10,))
        self.assertEqual(layer.shape_out, (4,))
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
