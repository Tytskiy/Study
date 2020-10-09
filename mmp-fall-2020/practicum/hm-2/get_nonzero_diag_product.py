import numpy as np


def get_nonzero_diag_product(X):
    diag = np.diag(X)
    if(not np.any(diag != 0)):
        return None
    return np.multiply.reduce(diag[diag != 0])
