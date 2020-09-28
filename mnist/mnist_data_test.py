import unittest
import numpy.testing as npt
from . import mnist_data

import os

import numpy as np

class MnistDataTestCase(unittest.TestCase):

    def testLoadMnist(self):
        X_train, X_test, y_train, y_test = mnist_data.load_mnist()
        self.assertEqual(X_train.shape, (60000, 28, 28))
        self.assertEqual(X_test.shape, (10000, 28, 28))
        self.assertEqual(y_train.shape, (60000,))
        self.assertEqual(y_test.shape, (10000,))

    def testPreprocessFlat(self):
        X_train, _, _, _ = mnist_data.load_mnist()
        X_train_pre = mnist_data.preprocess_flat(X_train)
        self.assertEqual(X_train_pre.shape, (60000, 784))
        self.assertEqual(np.min(X_train_pre), 0.0)
        self.assertEqual(np.max(X_train_pre), 1.0)

    def testPreprocessChannel(self):
        X_train, _, _, _ = mnist_data.load_mnist()
        X_train_pre = mnist_data.preprocess_channel(X_train)
        self.assertEqual(X_train_pre.shape, (60000, 28, 28, 1))
        self.assertEqual(np.min(X_train_pre), 0.0)
        self.assertEqual(np.max(X_train_pre), 1.0)

    def testToPred(self):
        Y_prob = np.array([
            [0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            [0.01, 0.05, 0.01, 0.01, 0.05, 0.01, 0.80, 0.05, 0.01, 0.05],
        ])
        y_pred = mnist_data.to_pred(Y_prob)
        expected_y_pred = np.array([1, 6])
        npt.assert_equal(y_pred, expected_y_pred)

    def testToOneHotProb(self):
        y_pred = np.array([2, 8])
        Y_prob = mnist_data.to_onehot_prob(y_pred)
        expected_Y_prob = np.array([
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        ])
        npt.assert_equal(Y_prob, expected_Y_prob)

    def testScorePredictions(self):
        _, y_true, y_pred, _ = self._loadPredictions()
        score, acc, cm = mnist_data.score_predictions(y_true, y_pred)
        self.assertEqual(score, 0.9258)
        expected_acc = np.array([
            0.9786, 0.9789, 0.8973, 0.9079, 0.9308,
            0.8778, 0.9520, 0.9241, 0.8840, 0.9167,
        ])
        npt.assert_almost_equal(acc, expected_acc, decimal=4)
        self.assertEqual(cm[0, 6], 5)
        self.assertEqual(cm[2, 2], 926)
        self.assertEqual(cm[9, 8], 7)

    def _loadPredictions(self):
        filename = self._getTestData("predictions.npz")
        pred_dict = np.load(filename)
        return [pred_dict[key] for key in ["X", "y_true", "y_pred", "Y_prob"]]

    def _getTestData(self, filename):
        dirname = os.path.dirname(__file__)
        return os.path.join(dirname, "testdata", filename)
