import numpy as np


def calc_expectations(h, w, X, Q):
    E = np.apply_along_axis(np.convolve, 1, Q, np.ones(w))[:, :Q.shape[1]]
    E = np.apply_along_axis(np.convolve, 0, E, np.ones(h))[:E.shape[0], :]
    E = E * X
    return E
