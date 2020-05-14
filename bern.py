from utils import docs
from pathlib import Path
import math
import numpy as np


def tokens(filepath: Path):
    with open(filepath) as f:
        return set(t for t in f.read().strip().split() if t.isalnum())


def index_word_appearancefraction():
    vocabulary = set()
    sets = (list(), list())
    for i, feel in enumerate(["pos", "neg"]):
        for doc in docs(feel=feel):
            toks = tokens(doc)
            vocabulary |= toks
            sets[i].append(toks)
    return {
        "index": {
            word: (
                sum(1 for s in sets[0] if word in s),
                sum(1 for s in sets[1] if word in s)
            ) for word in vocabulary},
        "total_nof_docs": tuple([len(s) for s in sets]),
    }


class ClassifierBernoulli:
    def __init__(self):
        self.d = index_word_appearancefraction()
        self.index = self.d["index"]

    def is_pos(self, filepath, alpha=1):
        assert 0 <= alpha <= 1
        toks = tokens(filepath) & self.d["index"].keys()

        feel_probs = []
        for i, feel in enumerate(["pos", "neg"]):
            ins = np.array([self.index[word][i] for word in toks], dtype=float)
            if alpha == 0 and min(ins) == 0:
                feel_probs.append(-np.inf)
                continue
            out_words = self.d["index"].keys() - toks
            outs = np.array([self.index[word][i] for word in out_words], dtype=float)
            if alpha == 0 and min(outs) == 0:
                feel_probs.append(-np.inf)
                continue
            nof_docs_feel = self.d["total_nof_docs"][i]

            ins = (ins+alpha) / (nof_docs_feel + 2*alpha)
            outs = 1 - ((outs+alpha) / (nof_docs_feel + 2*alpha))

            feel_probs.append(
                np.log(nof_docs_feel/sum(self.d["total_nof_docs"])) +
                np.sum(np.log(ins)) + np.sum(np.log(outs))
            )
        is_pos = feel_probs[0] >= feel_probs[1]
        return is_pos if feel_probs[0] != -math.inf else None
