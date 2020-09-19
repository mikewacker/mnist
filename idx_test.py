import unittest
import idx

import gzip
import struct
import tempfile

import numpy as np

class IdxTestCase(unittest.TestCase):

    def testReadArray(self):
        self._testReadArray(open)

    def testReadArray_GZip(self):
        self._testReadArray(gzip.open)

    def _testReadArray(self, open_fn):
        with tempfile.NamedTemporaryFile() as tmp:
            with open_fn(tmp.name, "wb") as f:
                self._writeSampleArray(f)
            with open_fn(tmp.name, "rb") as f:
                X = idx.read_array(f)
        self._verifySampleArray(X)

    def _writeSampleArray(self, f):
        header = struct.pack(">HBBII", 0, 0x0D, 2, 2, 3)
        f.write(header)
        data = struct.pack(">ffffff", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        f.write(data)

    def _verifySampleArray(self, X):
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(X.shape, (2, 3))
        X_expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertTrue((X == X_expected).all())
