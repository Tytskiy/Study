class Polynomial:
    def __init__(self, *args):
        i = len(args) - 1
        while args[i] == 0 and i >= 0:
            i -= 1
        self._coefs = args[: i + 1]

    def __call__(self, x):
        pows = [x] * len(self._coefs)
        pows = map(lambda x: x[1] ** x[0], enumerate(pows))
        return sum([x[0] * x[1] for x in zip(pows, self._coefs)])
