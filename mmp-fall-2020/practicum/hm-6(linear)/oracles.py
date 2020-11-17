import numpy as np
from scipy.special import expit


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """

    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """
        self.l2 = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        line_y = y.ravel()

        reg = self.l2*0.5*np.inner(w.ravel(), w.ravel())

        res = np.logaddexp(np.array([0]), -line_y*X.dot(w[:, None].ravel()))
        return reg + np.sum(res)/line_y.size

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        line_y = y.ravel()
        grad_reg = self.l2*w.ravel()
        if isinstance(X, np.ndarray):
            grad_res = self._mul(self._mul(X, line_y), expit(-line_y * X.dot(w[:, None]).ravel()))
        else:
            grad_res = self._mul(self._mul(X, line_y),
                                 expit(-line_y * X.dot(w[:, None]).ravel())).toarray()
        return grad_reg-np.sum(grad_res, axis=0)/line_y.size

    def _mul(self, X, y):
        if isinstance(X, np.ndarray):
            return (y[:, None]*X)
        else:
            return X.multiply(y[:, None])
