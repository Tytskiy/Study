from itertools import combinations


class WordContextGenerator:
    def __init__(self, s, k):
        self.s = s
        self.k = k

    def generate(self, s, k):
        for i in range(0, len(s)-k+1):
            for comb in combinations(s[i:i+k], 2):
                yield comb

    def __iter__(self):
        return self.generate(self.s, self.k)


for i in WordContextGenerator("мама очень хорошо мыла красивую раму".split(" "), 3):
    print(i)
