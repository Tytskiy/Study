class CooSparseMatrix:
    def _check_valid(self, a, shape):
        if (
            len(shape) != 2 or
            not isinstance(shape[0], int) or
            not isinstance(shape[0], int)
        ):
            return False
        for e in a:
            if (
                len(e) != 3 or
                e[0] >= shape[0] or
                e[1] >= shape[1] or
                e[0] < 0 or
                e[1] < 0 or
                not isinstance(e[0], int) or
                not isinstance(e[1], int)
            ):
                return False
        return True

    def __init__(self, ijx_list, shape):
        if not self._check_valid(ijx_list, shape):
            raise TypeError

        tmp_list = list(filter(lambda x: x[2] != 0, ijx_list))
        self._non_zero = {(i, j): x for i, j, x in tmp_list}
        if(len(self._non_zero) != len(tmp_list)):
            raise TypeError
        self.shape = shape

    def __getitem__(self, key):
        if (
            isinstance(key, tuple) and
            len(key) == 2 and
            (0 <= key[0] < self.shape[0]) and
            (0 <= key[1] < self.shape[1])
        ):
            return self._non_zero.get(key, 0)
        if isinstance(key, int) and (0 <= key < self.shape[0]):
            rg = range(self.shape[1])
            return CooSparseMatrix(
                [(key, i, self._non_zero.get((key, i), 0)) for i in rg],
                (1, self.shape[1]))
        raise TypeError

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            if(value != 0):
                self._non_zero[key] = value
            else:
                self._non_zero.pop(key, None)
        else:
            raise TypeError

    def __add__(self, other):
        if(not isinstance(other, CooSparseMatrix)):
            raise TypeError
        if(self.shape != other.shape):
            raise TypeError
        tmp = []
        tmp_keys = set(list(self._non_zero) +
                       list(other._non_zero))
        for i in tmp_keys:
            value = self[i[0], i[1]]+other[i[0], i[1]]
            if(value != 0):
                tmp.append(((i[0], i[1], value)))
        return CooSparseMatrix(tmp, self.shape)

    def __sub__(self, other):
        if(not isinstance(other, CooSparseMatrix)):
            raise TypeError
        if(self.shape != other.shape):
            raise TypeError
        tmp = []
        tmp_keys = set(list(self._non_zero) +
                       list(other._non_zero))
        for i in tmp_keys:
            value = self[i[0], i[1]]-other[i[0], i[1]]
            if(value != 0):
                tmp.append(((i[0], i[1], value)))
        return CooSparseMatrix(tmp, self.shape)

    def __mul__(self, coef):
        if(not isinstance(coef, (int, float))):
            raise TypeError
        if coef == 0:
            return CooSparseMatrix([], self.shape)
        tmp = []
        for k in self._non_zero:
            value = coef * self[k[0], k[1]]
            tmp.append((k[0], k[1], value))
        return CooSparseMatrix(tmp, self.shape)

    def __rmul__(self, coef):
        if(not isinstance(coef, (int, float))):
            raise TypeError
        if coef == 0:
            return CooSparseMatrix([], self.shape)
        tmp = []
        for k in self._non_zero:
            value = coef * self[k[0], k[1]]
            tmp.append((k[0], k[1], value))
        return CooSparseMatrix(tmp, self.shape)

    def __setattr__(self, attr, value):
        if(attr == "shape"):
            if(isinstance(value, tuple) and
               len(value) == 2 and isinstance(value[0], int) and
               isinstance(value[0], int)):
                if(not self.__dict__.get(attr)):
                    self.__dict__[attr] = value
                elif (self.shape[0]*self.shape[1] == value[0]*value[1]):
                    tmp = {}
                    for k in self._non_zero:
                        abs_pos = k[0]*self.shape[1]+k[1]
                        new_pos = (abs_pos//value[1], abs_pos % value[1])
                        tmp[new_pos] = self._non_zero[k]
                    self.__dict__[attr] = value
                    self._non_zero = tmp
                else:
                    raise TypeError
            else:
                raise TypeError
        elif(attr == "T"):
            raise AttributeError
        else:
            self.__dict__[attr] = value

    def __getattr__(self, attr):
        if(attr == "T"):
            tmp_shape = self.shape[::-1]
            tmp_non_zero = {(j, i, self[i, j]) for i, j in self._non_zero}
            return CooSparseMatrix(tmp_non_zero, tmp_shape)
        else:
            return self.__dict[attr]
