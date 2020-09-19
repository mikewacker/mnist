import gzip
import struct

import numpy as np

_DTYPE_TABLE = {
    0x08: np.uint8,
    0x09: np.int8,
    0x0B: np.int16,
    0x0C: np.int32,
    0x0D: np.float32,
    0x0E: np.float64,
}

def read_array(f):
    """Reads a numpy array from a file object for an IDX file."""
    dtype = _parse_dtype(f)
    shape = _parse_shape(f)
    return _parse_data(f, dtype, shape)

def _parse_dtype(f):
    """Parses the dtype of the data."""
    _, bcode = struct.unpack(">HB", f.read(3))
    dtype = _DTYPE_TABLE.get(bcode, None)
    if not dtype:
        msg = "unknown byte code for dtype: 0x{:02X}".format(bcode)
        raise ValueError(msg)
    return dtype

def _parse_shape(f):
    """Parses the shape of the data."""
    rank, = struct.unpack(">B", f.read(1))
    return tuple(
        struct.unpack(">I", f.read(4))[0]
        for _ in range(rank))

def _parse_data(f, dtype, shape):
    """Parses the data."""
    dtype_big = np.dtype(dtype).newbyteorder(">")
    count = np.prod(np.array(shape))
    # See: https://github.com/numpy/numpy/issues/13470
    use_buffer = type(f) == gzip.GzipFile
    if use_buffer:
        data = np.frombuffer(f.read(), dtype_big, count)
    else:
        data = np.fromfile(f, dtype_big, count)
    return data.astype(dtype).reshape(shape)
