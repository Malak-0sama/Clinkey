"""Compact BM25 (Okapi) implementation — no external dependency."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.doc_tokens = [self._tok(d) for d in corpus]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.df: Counter = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] += 1
        self.idf = {
            t: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()
        }

    @staticmethod
    def _tok(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def scores(self, query: str) -> list[float]:
        q_tokens = self._tok(query)
        if not q_tokens or self.N == 0:
            return [0.0] * self.N
        scores = [0.0] * self.N
        qf = Counter(q_tokens)
        for term, qt in qf.items():
            idf = self.idf.get(term)
            if idf is None or idf == 0:
                continue
            for i, toks in enumerate(self.doc_tokens):
                tf = toks.count(term)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom * (qt + 0.0)
        return scores