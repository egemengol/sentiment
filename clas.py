from utils import docs
from pathlib import Path
from collections import Counter
import math
import numpy as np


def tokenize_classic(filepath: Path):
    with open(filepath) as f:
        return Counter(t for t in f.read().strip().split() if t.isalnum()  )# and t not in excluded)


def tokenize_binary(filepath: Path):
    with open(filepath) as f:
        return Counter(set(t for t in f.read().strip().split() if t.isalnum()))


def vocabulary(tokenize):
    vocab = {
        "pos": Counter(),
        "neg": Counter(),
    }
    for feel, counter in vocab.items():
        for doc in docs("train", feel):
            counter += tokenize(doc)
    return vocab


class ClassifierClassic:
    def __init__(self, tokenize):
        self.tokenize = tokenize
        nof_docs = {
            "pos": sum(1 for _ in docs("train", "pos")),
            "neg": sum(1 for _ in docs("train", "neg")),
        }
        nof_docs_total = sum(nof_docs.values())

        voc = vocabulary(self.tokenize)

        self.dict = {"pos": {"counter": voc["pos"]}, "neg": {"counter": voc["neg"]}}

        for feel in self.dict:
            self.dict[feel]["nof_tokens"] = sum(self.dict[feel]["counter"].values())
            self.dict[feel]["log_prob"] = math.log(nof_docs[feel] / nof_docs_total)

        set_pos = set(self.dict["pos"]["counter"])
        set_neg = set(self.dict["neg"]["counter"])
        set_total = set_pos | set_neg

        self.dict["voc_size"] = len(set_total)

    def is_pos(self, filepath, alpha=1):
        toks = self.tokenize(filepath)

        feel_probs = []
        for feel in ["pos", "neg"]:
            probs = list()
            for word, count in toks.items():
                prob = self.dict[feel]["counter"][word]
                if prob == 0 and self.dict["pos"]["counter"][word] and self.dict["neg"]["counter"][word]:
                    continue
                probs.extend([prob] * count)

            if alpha == 0:
                probs = [p for p in probs if p != 0]

            probs = np.array(probs, dtype=float)

            nof_tokens = self.dict[feel]["nof_tokens"]
            smoothing = alpha * self.dict["voc_size"]
            probs = (probs+alpha) / (nof_tokens + smoothing)
            log_prob = self.dict[feel]["log_prob"]
            feel_probs.append(log_prob + np.sum(np.log(probs)))
        is_pos = feel_probs[0] > feel_probs[1]
        return is_pos if feel_probs[0] != -math.inf else None