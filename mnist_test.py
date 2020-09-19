import unittest
import mnist

class MnistTestCase(unittest.TestCase):

    def testLoadMnist(self):
        X_train, X_test, y_train, y_test = mnist.load_mnist()
        self.assertEqual(X_train.shape, (60000, 28, 28))
        self.assertEqual(X_test.shape, (10000, 28, 28))
        self.assertEqual(y_train.shape, (60000,))
        self.assertEqual(y_test.shape, (10000,))
