import scipy.sparse as spr

import numpy as np
from oracles import BinaryLogistic
import time
from scipy.special import expit

loss_functions = {
    "binary_logistic": BinaryLogistic
}


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function="binary_logistic", step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, fit_intercept=False, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss = loss_functions[loss_function](kwargs.get("l2_coef", 1))
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.history = {"time": [], "func": []}

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """

        if w_0 is None:
            w = np.zeros(X.shape[1])
        else:
            w = w_0

        if self.fit_intercept:
            w = np.hstack((w, [1]))
            if isinstance(X, np.ndarray):
                X = np.hstack((X, np.ones(X.shape[0])))
            else:
                X = spr.hstack((X, np.ones(X.shape[0])[:, None]), format="csr")

        n = self.step_alpha

        prev_func = 0
        curr_func = self.loss.func(X, y, w)
        self.history["func"].append(curr_func)
        self.history["time"].append(0)

        for i in range(self.max_iter):
            if np.abs(curr_func - prev_func) < self.tolerance:
                break

            start_time = time.time()
            w = w - n*self.loss.grad(X, y, w, intercept=self.fit_intercept)
            self.history["time"].append(time.time()-start_time)

            n /= (i+1)**self.step_beta

            prev_func = curr_func
            curr_func = self.loss.func(X, y, w)
            self.history["func"].append(curr_func)

        self.w = w
        if trace:
            return self.history

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        if self.fit_intercept:
            if isinstance(X, np.ndarray):
                X = np.hstack((X, np.ones(X.shape[0])))
            else:
                X = spr.hstack((X, np.ones(X.shape[0])[:, None]), format="csr")

        tmp = X.dot(self.w[:, None]).ravel() > 0
        return np.array(np.array([1 if i == True else -1 for i in tmp]))

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        if self.fit_intercept:
            if isinstance(X, np.ndarray):
                X = np.hstack((X, np.ones(X.shape[0])))
            else:
                X = spr.hstack((X, np.ones(X.shape[0])[:, None]), format="csr")

        tmp = expit(X.dot(self.w[:, None])).reshape(-1, 1)
        return np.hstack((1-tmp, tmp))

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.loss(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.loss(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function="binary_logistic", batch_size=1000, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=10000, random_seed=153, fit_intercept=False, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход


        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """
        super().__init__(loss_function=loss_function, step_alpha=step_alpha, step_beta=step_beta,
                         tolerance=tolerance, max_iter=max_iter, fit_intercept=fit_intercept, **kwargs)
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.history["epoch_num"] = []
        self.history["weights_diff"] = []

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """

        if w_0 is None:
            w = np.zeros(X.shape[1])
        else:
            w = w_0

        if self.fit_intercept:
            w = np.hstack((w, [1]))
            if isinstance(X, np.ndarray):
                X = np.hstack((X, np.ones(X.shape[0])))
            else:
                X = spr.hstack((X, np.ones(X.shape[0])[:, None]), format="csr")

        n = self.step_alpha

        prev_func = 0
        curr_func = self.loss.func(X, y, w)
        prev_epoch = 0
        curr_epoch = 0
        self.history["func"].append(curr_func)
        self.history["time"].append(0)
        self.history['epoch_num'].append(curr_epoch)
        self.history['weights_diff'].append(np.linalg.norm(w))

        np.random.seed(self.random_seed)
        batchs = np.vstack([np.random.choice(X.shape[0], self.batch_size, replace=False)
                            for i in range(self.max_iter)])
        start_time = time.time()
        for i in range(self.max_iter):
            if np.abs(curr_func - prev_func) < self.tolerance:
                break

            curr_epoch = i*self.batch_size/X.shape[0]
            if(curr_epoch-prev_epoch > log_freq and trace):
                self.history["time"].append(time.time()-start_time)
                start_time = time.time()
                prev_func = curr_func
                curr_func = self.loss.func(X, y, w)
                self.history["func"].append(curr_func)
                self.history['weights_diff'].append(np.linalg.norm(w))
                self.history['epoch_num'].append(curr_epoch)
            w = w - n*self.loss.grad(X[batchs[i]], y[batchs[i]], w, intercept=self.fit_intercept)
            n /= (i+1)**self.step_beta

        self.w = w
        if trace:
            return self.history
