import numpy as np


def euclidean_distance(X, Y):
    X_square = (X*X).sum(axis=1).reshape(-1, 1)
    Y_square = (Y*Y).sum(axis=1).reshape(1, -1)
    ans = (X_square+Y_square-2*np.dot(X, Y.T))**0.5
    return ans


def cosine_distance(X, Y):
    norm_X = np.linalg.norm(X, axis=1).reshape(-1, 1)
    norm_Y = np.linalg.norm(Y, axis=1).reshape(1, -1)
    ans = 1-np.dot(X, Y.T)/norm_X/norm_Y
    return ans
