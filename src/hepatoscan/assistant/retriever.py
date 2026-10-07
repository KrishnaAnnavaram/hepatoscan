"""BM25 retrieval over knowledge-base passages (pure Python)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .kb import Passage

STOP = frozenset("a an and are as at be by can do does for from has have how i if in is it its me my of on or "
                 "the this to was what when which who why will with you your".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


class BM25:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75) -> None:
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens(p.title + " " + p.text)) for p in passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(passages)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3, min_score: float = 0.5) -> list[Hit]:
        q = tokens(query)
        scored = []
        for i, doc in enumerate(self.docs):
            s = 0.0
            for t in q:
                if t not in doc:
                    continue
                tf = doc[t]
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg))
            if s >= min_score:
                scored.append(Hit(self.passages[i], s))
        scored.sort(key=lambda h: (-h.score, h.passage.pid))
        return scored[:k]
