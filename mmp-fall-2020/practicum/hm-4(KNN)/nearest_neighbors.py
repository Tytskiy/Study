import numpy as np
import distances
from sklearn.neighbors import NearestNeighbors


class NearestNeighborsCustom:
    distance = {
        "euclidean": distances.euclidean_distance,
        "cosine": distances.cosine_distance
    }

    def __init__(self, metric="euclidean"):
        self.metric = metric

    def fit(self, X):
        self.X = X

    def kneighbors(self, X, n_neighbors=5,
                   return_distance=True, kind="mergesort"):

        mat_pairwise = NearestNeighborsCustom.distance[self.metric](X, self.X)
        arg_sort = np.argsort(mat_pairwise, axis=1, kind=kind)

        mat_pairwise = np.sort(mat_pairwise, axis=1, kind=kind)
        mat_pairwise = mat_pairwise[:, : n_neighbors]

        if return_distance:
            return mat_pairwise, arg_sort[:, :n_neighbors]
        return arg_sort[:, :n_neighbors]


class KNNClassifier:
    def __init__(self, k=5, strategy="my_own",
                 metric="euclidean", weights=False, test_block_size=None):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.block_size = test_block_size

    def fit(self, X, y):
        if self.strategy == "my_own":
            self.NN = NearestNeighborsCustom(metric=self.metric)
        else:
            self.NN = NearestNeighbors(
                n_neighbors=self.k, algorithm=self.strategy,
                metric=self.metric)
        self.NN.fit(X)
        self.y = y

    def find_kneighbors(self, X, return_distance=True):
        if self.block_size is None:
            return self.NN.kneighbors(
                X, n_neighbors=self.k, return_distance=return_distance)

        mat_indexs = np.empty((X.shape[0], self.k))
        if return_distance:
            mat_pairwise = np.empty((X.shape[0], self.k))

        # люблю --max_length_size=79 pep8
        love_pep = self.block_size
        for i in range(love_pep, X.shape[0] + love_pep, love_pep):

            if return_distance:
                tmp1, tmp2 = self.NN.kneighbors(
                    X[i - love_pep: i, :],
                    n_neighbors=self.k, return_distance=return_distance)
                mat_pairwise[i - love_pep: i, :] = tmp1
                mat_indexs[i - love_pep: i, :] = tmp2
            else:
                tmp = self.NN.kneighbors(
                    X[i - love_pep: i, :],
                    n_neighbors=self.k, return_distance=return_distance)
                mat_indexs[i - love_pep: i, :] = tmp

        if return_distance:
            return mat_pairwise, mat_indexs.astype(int)
        return mat_indexs.astype(int)

    def predict(self, X):
        if self.weights:
            mat_pairwise, mat_indexs = self.find_kneighbors(X)
            mat_weights = 1/(mat_pairwise+10e-5)
            # После этого задания любая строчка
            # кода мне кажется неоптимальной по памяти
            del mat_pairwise

            labels = np.apply_along_axis(
                lambda x: self.y[x], arr=mat_indexs, axis=1)

            uniq_y = np.unique(self.y)
            tmp_tensor = labels[:, :, np.newaxis] == uniq_y.reshape(1, 1, -1)
            tmp_tensor = tmp_tensor*mat_weights[:, :, np.newaxis]

            predict = uniq_y[tmp_tensor.sum(axis=1).argmax(axis=1)]
        else:
            mat_indexs = self.find_kneighbors(X, return_distance=False)
            labels = np.apply_along_axis(
                lambda x: self.y[x], arr=mat_indexs, axis=1)
            predict = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=1, arr=labels)
        return predict
