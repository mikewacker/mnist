import unittest
import nn_layers

import numpy as np

class NNLayersTestCase(unittest.TestCase):

    def testSigmoid(self):
        log_odds = np.log(np.array([[3.0, 4.0], [1.0, 0.25]]))
        A = nn_layers._sigmoid(log_odds)
        expected_prob = np.array([[0.75, 0.80], [0.50, 0.20]])
        self.assertArraysEqual(A, expected_prob)

    def testSoftmax(self):
        log_odds = np.log(np.array([[1.0, 3.0, 1.0], [2.0, 5.0, 3.0]]))
        A = nn_layers._softmax(log_odds)
        expected_prob = np.array([[0.2, 0.6, 0.2], [0.2, 0.5, 0.3]])
        self.assertArraysEqual(A, expected_prob)

    def assertArraysEqual(self, A, expected_A):
        A_diff = np.abs(A - expected_A)
        self.assertLess(np.max(A_diff), 1e-8)
