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
