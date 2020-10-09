import numpy as np


def get_max_before_zero(X):
    mask = X == 0
    if(np.any(mask)):
        mask = np.concatenate((np.array([False]), mask), axis=0)
        ans = X[mask[: -1]]
        if ans.size != 0:
            return ans.max()
    return None
