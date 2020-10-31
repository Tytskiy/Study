import numpy as np


class RleSequence:
    def __init__(self, seq):
        self.elems, self.count = self.encode_rle(seq)
        self.size = seq.size
        if(seq.size != 0):
            self.curr_elem = self.elems[0]
        else:
            self.curr_elem = None
        self.curr_counts = 1
        self.i = 0

    def encode_rle(self, seq):
        if(seq.size == 0):
            return np.array([]), np.array([])
        bias = np.append(seq[1:], seq[-1])
        mask = (bias != seq)
        mask[-1] = True
        tmp = np.where(mask)[0]
        tmp[1:] -= tmp[:-1]
        tmp[0] += 1
        return seq[mask], tmp

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.count.size:
            raise StopIteration
        tmp = self.curr_elem
        if self.curr_counts >= self.count[self.i]:
            self.i += 1
            if self.i < self.count.size:
                self.curr_counts = 1
                self.curr_elem = self.elems[self.i]
        else:
            self.curr_counts += 1
        return tmp

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key = self.size+key
            if(key < 0 or key > self.size):
                raise IndexError
            tmp = 0
            i = 0
            while(tmp < key+1 and i < self.count.size):
                tmp += self.count[i]
                i += 1
            return self.elems[i-1]
        if isinstance(key, slice):
            tmp_curr_count = self.curr_counts
            tmp_curr_elem = self.curr_elem
            tmp_i = self.i
            prev_i = 0
            slice_array = []
            for i in range(*key.indices(self.size)):
                for _ in range(i-prev_i):
                    next(self)
                slice_array.append(self.curr_elem)
                prev_i = i
            self.curr_counts = tmp_curr_count
            self.curr_elem = tmp_curr_elem
            self.i = tmp_i
            return np.array(slice_array)

    def __contains__(self, target):
        if self.elems[self.elems == target].size == 0:
            return False
        return True

