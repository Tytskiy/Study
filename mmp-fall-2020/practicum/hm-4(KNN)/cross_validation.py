import numpy as np
import nearest_neighbors as nn


scores = {
    "accuracy": lambda X, Y: (X == Y).mean()
}


def kfold(n, n_fold):
    if(n < n_fold or n_fold < 2):
        raise ValueError

    r = n % n_fold
    a = n // n_fold

    size_folds = np.array([a + 1]*r+[a]*(n_fold - r))
    splits_folds = np.cumsum(size_folds)
    folds = np.split(np.arange(n), splits_folds)

    index_folds = []
    for i in range(n_fold-1, -1, -1):
        index_folds.append((np.hstack(folds[0:i]+folds[i+1:]), folds[i]))
    return index_folds


def knn_cross_val_score(X, y, k_list, cv=None, score="accuracy", **kwargs):
    if cv is None:
        cv = kfold(X.shape[0], 2)
    model = nn.KNNClassifier(k_list[-1], **kwargs)

    k_neigh = {}
    for train, test in cv:
        model.fit(X[train], y[train])

        if kwargs.get("weights", False):
            mat_pairwise, mat_indexs = model.find_kneighbors(X[test])
            mat_weights = 1/(mat_pairwise+10e-5)
        else:
            mat_indexs = model.find_kneighbors(X[test], return_distance=False)

        labels = np.apply_along_axis(lambda x: y[train][x], arr=mat_indexs, axis=1)
        for k in k_list:
            if k_neigh.get(k, None) is None:
                k_neigh[k] = []

            uniq_y = np.unique(y[train])
            tmp = (labels[:, :k, np.newaxis] == uniq_y.reshape(1, 1, -1))

            if kwargs.get("weights", False):
                tmp = tmp*mat_weights[:, :k, np.newaxis]

            predict = uniq_y[tmp.sum(axis=1).argmax(axis=1)]
            k_neigh[k].append(scores[score](predict, y[test]))

    return k_neigh
