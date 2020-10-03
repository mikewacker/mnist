import unittest
import numpy.testing as npt
from .nn import NeuralNetwork

import os
import tempfile

import numpy as np

from . import nn_layers
from . import nn_optimizers

class NNTestCase(unittest.TestCase):

    def testNeuralNetwork(self):
        nn = self._initNeuralNetwork()
        self.assertEqual(nn.num_params, 266610)
        X_train, X_test, y_train, y_test = self._loadMnist1000()

        nn.train(
            X_train, y_train, num_epochs=2,
            learning_rate=0.001, minibatch_size=64, weight_decay=0.0001)
        y_pred, _ = nn.predict(X_test)

        acc = np.mean(y_test == y_pred)
        self.assertGreater(acc, 0.5)

    def testNeuralNetwork_CompareToGolden(self):
        nn = self._initNeuralNetwork()
        X_train, X_test, y_train, y_test = self._loadMnist1000()

        nn.train(
            X_train, y_train, num_epochs=2,
            learning_rate=0.001, minibatch_size=64, weight_decay=0.0001)
        _, Y_prob = nn.predict(X_test)

        expected_Y_prob = self._loadGoldenYProb()
        npt.assert_equal(Y_prob, expected_Y_prob)

    def testNeuralNetwork_Persistence(self):
        nn1 = self._initNeuralNetwork()
        nn2 = self._initNeuralNetwork()
        X_train, X_test, y_train, y_test = self._loadMnist1000()

        nn1.train(
            X_train, y_train,
            learning_rate=0.001, minibatch_size=64, weight_decay=0.0001)
        with tempfile.NamedTemporaryFile() as tmp:
            with open(tmp.name, "wb") as f:
                nn1.save(f)
            with open(tmp.name, "rb") as f:
                nn2.load(f)

        np.random.seed(875659)
        nn1.train(
            X_train, y_train,
            learning_rate=0.001, minibatch_size=64, weight_decay=0.0001)
        y_pred1, Y_prob1 = nn1.predict(X_test)
        cost1 = nn1.cost(y_test)

        np.random.seed(875659)
        nn2.train(
            X_train, y_train,
            learning_rate=0.001, minibatch_size=64, weight_decay=0.0001)
        y_pred2, Y_prob2 = nn2.predict(X_test)
        cost2 = nn2.cost(y_test)

        npt.assert_equal(y_pred1, y_pred2)
        npt.assert_equal(Y_prob1, Y_prob2)
        self.assertEqual(cost1, cost2)

    def _initNeuralNetwork(self):
        np.random.seed(744502)
        return NeuralNetwork(
            preprocess_fn=self._preprocessFlat,
            hidden_layers=[
                nn_layers.dense(784, 300),
                nn_layers.dense(300, 100),
            ],
            output_layer=nn_layers.multiclass_output(100, 10),
            optimizer=nn_optimizers.adam())

    def _preprocessFlat(self, X):
        X = X.reshape(-1, 784)
        return X / 255

    def _loadMnist1000(self):
        filename = self._getTestData("mnist1000.npz")
        data = np.load(filename)
        return [data[key] for key in ["X_train", "X_test", "y_train", "y_test"]]

    def _loadGoldenYProb(self):
        filename = self._getTestData("Y_prob.npy")
        return np.load(filename)

    def _getTestData(self, filename):
        dirname = os.path.dirname(__file__)
        return os.path.join(dirname, "testdata", filename)
