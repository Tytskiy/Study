import numpy as np


def encode_rle(X):
    bias = np.append(X[1:], X[-1])
    mask = (bias != X)
    mask[-1] = True
    tmp = np.where(mask)[0]
    tmp[1:] -= tmp[:-1]
    tmp[0] += 1
    return X[mask], tmp
