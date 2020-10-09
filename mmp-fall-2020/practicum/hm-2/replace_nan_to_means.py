import numpy as np


def replace_nan_to_means(X):
    Y = X.copy()
    mean = np.nanmean(Y, axis=0)
    mask = np.isnan(Y)
    Y = np.nan_to_num(Y, 0) + mask * np.nan_to_num(mean, 0)[np.newaxis, :]
    return Y
