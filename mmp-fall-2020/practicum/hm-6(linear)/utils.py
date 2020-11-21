import numpy as np
def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    e = np.zeros(w.size)
    result = np.zeros(w.size)
    for i in range(w.size):
        e[i] = 1
        result[i] = (function(w+eps*e)-function(w))/eps
        e[i] = 0
    return result

