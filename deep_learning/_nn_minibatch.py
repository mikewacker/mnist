import numpy as np

def minibatches(X, Y, size=0):
    """Generates the mini-batches, also resampling to align the sizes."""
    size = size or X.shape[0]
    X_shuffle, Y_shuffle = _shuffle(X, Y, size)
    m = X_shuffle.shape[0]
    offset = 0
    while offset < m:
        end = offset + size
        X_mb = X_shuffle[offset:end]
        Y_mb = Y_shuffle[offset:end]
        yield X_mb, Y_mb
        offset = end

def _shuffle(X, Y, size):
    """Shuffles the samples, also resampling to align the sizes."""
    m = X.shape[0]
    if size >= m:
        return X, Y

    # Initialize the empty shuffled data.
    num_mbs = int(np.ceil(m / size))
    m_align = num_mbs * size
    X_shuffle = np.empty((m_align, *X.shape[1:]))
    Y_shuffle = np.empty((m_align, *Y.shape[1:]))

    # Shuffle.
    indices = np.random.permutation(m)
    X_shuffle[:m] = X[indices]
    Y_shuffle[:m] = Y[indices]

    # Resample to align the number of samples with the mini-batch size.
    if m_align == m:
        return X_shuffle, Y_shuffle
    extra = m_align - m
    indices = np.random.permutation(m)[:extra]
    X_shuffle[m:] = X[indices]
    Y_shuffle[m:] = Y[indices]
    return X_shuffle, Y_shuffle
