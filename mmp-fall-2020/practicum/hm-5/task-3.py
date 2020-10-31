import random
import copy
import numpy as np


class BatchGenerator:
    def __init__(self, seq, batch_size, shuffle=False):
        self.seq = copy.deepcopy(seq)
        self.batch_size = batch_size
        self.shuffle = shuffle
        if shuffle:
            self.random(self.seq)

    def generate(self, seq, batch_size):
        curr = 0
        if isinstance(seq, np.ndarray):
            while curr*batch_size < seq.shape[1]:
                yield seq[:, curr*batch_size:(curr+1)*batch_size]
                curr += 1
        elif isinstance(seq, list):
            while curr*batch_size < len(seq[0]):
                yield [elem[curr*batch_size:(curr+1)*batch_size] for elem in seq]
                curr += 1

    def random(self, seq):
        if isinstance(seq, np.ndarray):
            np.apply_along_axis(np.random.shuffle, arr=seq, axis=0)
            return seq
        elif isinstance(seq, list):
            [random.shuffle(e) for e in seq]
            return seq
        else:
            raise TypeError

    def __iter__(self):
        return self.generate(self.seq, self.batch_size)
