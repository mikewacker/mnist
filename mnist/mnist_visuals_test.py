import unittest
from . import mnist_visuals

import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from . import mnist_data

class MnistVisualsTestCase(unittest.TestCase):

    def testShowImages(self):
        self._testShowImages("images.png")

    def testShowImages_Digits(self):
        self._testShowImages("images-digits.png", digits=[0,1])

    def testShowImages_Size(self):
        self._testShowImages("images-size.png", size=7)

    def _testShowImages(self, expected_path, **kwds):
        X_train, _, y_train, _ = mnist_data.load_mnist()
        np.random.seed(0)
        mnist_visuals.show_images(X_train, y_train, **kwds)
        self._compareToGoldenImage(expected_path)

    def testShowPerformance(self):
        _, y_true, y_pred, _ = self._loadPredictions()
        mnist_visuals.show_performance(y_true, y_pred)
        self._compareToGoldenImage("performance.png")

    def testShowPredictions(self):
        self._testShowPredictions("predictions.png")

    def testShowPredictions_Digits(self):
        self._testShowPredictions(
            "predictions-digits.png", true_digits=6, pred_digits=[0, 6])

    def testShowPredictions_Size(self):
        self._testShowPredictions("predictions-size.png", size=4)

    def _testShowPredictions(self, expected_path, **kwds):
        X, y_true, y_pred, Y_Prob = self._loadPredictions()
        np.random.seed(0)
        mnist_visuals.show_predictions(X, y_true, y_pred, Y_Prob, **kwds)
        self._compareToGoldenImage(expected_path)

    def _loadPredictions(self):
        filename = self._getTestData("predictions.npz")
        pred_dict = np.load(filename)
        return [pred_dict[key] for key in ["X", "y_true", "y_pred", "Y_prob"]]

    def _compareToGoldenImage(self, golden_filename):
        golden_filename = self._getTestData(golden_filename)
        with tempfile.NamedTemporaryFile() as tmp:
            plt.savefig(tmp.name, format="png")
            self.assertImagesEqual(tmp.name, golden_filename)

    def _getTestData(self, filename):
        dirname = os.path.dirname(__file__)
        return os.path.join(dirname, "testdata", filename)

    def assertImagesEqual(self, path1, path2):
        bytes1 = self._getImageBytes(path1)
        bytes2 = self._getImageBytes(path2)
        self.assertEqual(bytes1, bytes2)

    def _getImageBytes(self, path):
        with open(path, "rb") as f:
            return f.read()
