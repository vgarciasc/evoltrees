import cython
import numpy as np
cimport numpy as np

np.import_array()
@cython.wraparound(False)
@cython.boundscheck(False)

def get_leaf(np.ndarray[np.double_t, ndim=1] x,
                    np.ndarray[Py_ssize_t, ndim=1] attributes,
                    np.ndarray[np.double_t, ndim=1] thresholds,
                    int depth):

    cdef Py_ssize_t curr_depth = depth - 1
    cdef Py_ssize_t node_idx = 0
    cdef Py_ssize_t leaf_idx = 0

    while curr_depth >= 0:
        if x[attributes[node_idx]] <= thresholds[node_idx]:
            node_idx += 1
        else:
            node_idx += 2 ** curr_depth
            leaf_idx += 2 ** curr_depth
        curr_depth -= 1

    return leaf_idx

def get_leaves_dataset(np.ndarray[np.double_t, ndim=2] X,
                       np.ndarray[Py_ssize_t, ndim=1] attributes,
                       np.ndarray[np.double_t, ndim=1] thresholds,
                       int depth):

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]
    cdef Py_ssize_t n_leaves = 2 ** depth
    cdef Py_ssize_t i, j

    cdef np.ndarray[np.double_t, ndim=2] leaves = np.zeros((n_samples, n_leaves), dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] x = np.zeros(n_features, dtype=np.double)

    for i in range(n_samples):
        for j in range(n_features):
            x[j] = X[i, j]
        leaves[i, get_leaf(x, attributes, thresholds, depth)] = 1

    return leaves
