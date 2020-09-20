import unittest
import mnist

import tempfile

import numpy as np
import matplotlib.pyplot as plt

class MnistTestCase(unittest.TestCase):

    def testLoadMnist(self):
        X_train, X_test, y_train, y_test = mnist.load_mnist()
        self.assertEqual(X_train.shape, (60000, 28, 28))
        self.assertEqual(X_test.shape, (10000, 28, 28))
        self.assertEqual(y_train.shape, (60000,))
        self.assertEqual(y_test.shape, (10000,))

    def testPreprocessFlat(self):
        X_train, _, _, _ = mnist.load_mnist()
        X_train_pre = mnist.preprocess_flat(X_train)
        self.assertEqual(X_train_pre.shape, (60000, 784))
        self.assertEqual(np.min(X_train_pre), 0.0)
        self.assertEqual(np.max(X_train_pre), 1.0)

    def testPreprocessChannel(self):
        X_train, _, _, _ = mnist.load_mnist()
        X_train_pre = mnist.preprocess_channel(X_train)
        self.assertEqual(X_train_pre.shape, (60000, 28, 28, 1))
        self.assertEqual(np.min(X_train_pre), 0.0)
        self.assertEqual(np.max(X_train_pre), 1.0)

    def testToPred(self):
        Y_prob = np.array([
            [0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            [0.01, 0.05, 0.01, 0.01, 0.05, 0.01, 0.80, 0.05, 0.01, 0.05],
        ])
        y_pred = mnist.to_pred(Y_prob)
        self.assertTrue((y_pred == np.array([1, 6])).all())

    def testToOneHotProb(self):
        y_pred = np.array([2, 8])
        Y_prob = mnist.to_onehot_prob(y_pred)
        Y_prob_expected = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ])
        self.assertTrue((Y_prob == Y_prob_expected).all())

    def testShowImages(self):
        self._testShowImages("testdata/images.png")

    def testShowImages_Digits(self):
        self._testShowImages("testdata/images-digits.png", digits=[0,1])

    def testShowImages_Size(self):
        self._testShowImages("testdata/images-size.png", size=7)

    def _testShowImages(self, expected_path, **kwds):
        X_train, _, y_train, _ = mnist.load_mnist()
        np.random.seed(0)
        mnist.show_images(X_train, y_train, **kwds)
        self._saveAndCompareImage(expected_path)

    def testShowPredictions(self):
        self._testShowPredictions("testdata/predictions.png")

    def testShowPredictions_Digits(self):
        self._testShowPredictions(
            "testdata/predictions-digits.png",
            true_digits=6, pred_digits=[0, 6])

    def testShowPredictions_Size(self):
        self._testShowPredictions("testdata/predictions-size.png", size=4)

    def _testShowPredictions(self, expected_path, **kwds):
        pred_dict = np.load("testdata/predictions.npz")
        X, y_true, y_pred, Y_Prob = [
            pred_dict[key] for key in ["X", "y_true", "y_pred", "Y_prob"]]
        np.random.seed(0)
        mnist.show_predictions(X, y_true, y_pred, Y_Prob, **kwds)
        self._saveAndCompareImage(expected_path)

    def _saveAndCompareImage(self, expected_path):
        with tempfile.NamedTemporaryFile() as tmp:
            plt.savefig(tmp.name, format="png")
            self.assertImagesEqual(tmp.name, expected_path)

    def assertImagesEqual(self, path1, path2):
        bytes1 = self._getImageBytes(path1)
        bytes2 = self._getImageBytes(path2)
        self.assertEqual(bytes1, bytes2)

    def _getImageBytes(self, path):
        with open(path, "rb") as f:
            return f.read()
