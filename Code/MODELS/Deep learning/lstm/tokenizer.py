import numpy as np
from collections import Counter


class Tokenizer:

    def __init__(self, max_vocab=20000, max_len=100):
        self.max_vocab = max_vocab
        self.max_len   = max_len
        self.word2idx  = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word  = {}

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(str(text).split())

        for i, (word, _) in enumerate(counter.most_common(self.max_vocab - 2), start=2):
            self.word2idx[word] = i

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        print(f"[Tokenizer] Vocabulary size: {len(self.word2idx):,}")

    def encode(self, text):
        tokens = str(text).split()
        ids    = [self.word2idx.get(t, 1) for t in tokens]

        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return np.array(ids, dtype=np.int64)