import unittest
import numpy.testing as npt
import _nn_minibatch

import numpy as np

class NNMiniBatchTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(934310)

    def testMiniBatches(self):
        X = np.array([[3., 2.5], [1., 1.5], [2., 2.5], [4., 4.5]])
        Y = np.array([[3.], [1.], [2.], [4.]])
        num_mbs = 0
        checksum = 0.
        for X_mb, Y_mb in _nn_minibatch.minibatches(X, Y, 2):
            num_mbs += 1
            checksum += np.sum(Y_mb)
            self.assertEqual(X_mb.shape, (2, 2))
            self.assertEqual(Y_mb.shape, (2, 1))
            npt.assert_equal(X[:, 0:1], Y)
        self.assertEqual(num_mbs, 2)
        self.assertEqual(checksum, 10.)

    def testMiniBatches_AlignSize(self):
        X = np.array([[3., 2.5], [1., 1.5], [2., 2.5], [4., 4.5]])
        Y = np.array([[3.], [1.], [2.], [4.]])
        num_mbs = 0
        checksum = 0.
        for X_mb, Y_mb in _nn_minibatch.minibatches(X, Y, 3):
            num_mbs += 1
            checksum += np.sum(Y_mb)
            self.assertEqual(X_mb.shape, (3, 2))
            self.assertEqual(Y_mb.shape, (3, 1))
        self.assertEqual(num_mbs, 2)
        self.assertGreaterEqual(checksum, 13.)
        self.assertLessEqual(checksum, 17.)

    def testMiniBatches_SingleBatch(self):
        X = np.zeros((5, 10))
        Y = np.zeros((5, 4))
        num_mbs = 0
        for X_mb, Y_mb in _nn_minibatch.minibatches(X, Y):
            num_mbs += 1
            self.assertEqual(X_mb.shape, (5, 10))
            self.assertEqual(Y_mb.shape, (5, 4))
        self.assertEqual(num_mbs, 1)

    def testMiniBatches_SizeLargerThanSamples(self):
        X = np.zeros((5, 10))
        Y = np.zeros((5, 4))
        num_mbs = 0
        for X_mb, Y_mb in _nn_minibatch.minibatches(X, Y, 10):
            num_mbs += 1
            self.assertEqual(X_mb.shape, (5, 10))
            self.assertEqual(Y_mb.shape, (5, 4))
        self.assertEqual(num_mbs, 1)
