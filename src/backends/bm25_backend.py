"""BM25 retrieval backend using rank_bm25."""
from __future__ import annotations
from typing import List, Tuple
import math
from collections import Counter


class BM25Backend:
    """BM25 Okapi retrieval backend."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.idf: dict = {}
        self.avgdl: float = 0.0

    def fit(self, documents: List[str]) -> None:
        """Tokenise and index documents."""
        self.corpus = [doc.lower().split() for doc in documents]
        self.avgdl = sum(len(d) for d in self.corpus) / len(self.corpus)
        df: Counter = Counter()
        for doc in self.corpus:
            df.update(set(doc))
        N = len(self.corpus)
        self.idf = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Return top_k (doc_index, score) pairs."""
        tokens = query.lower().split()
        scores = []
        for idx, doc in enumerate(self.corpus):
            dl = len(doc)
            freq_map = Counter(doc)
            score = sum(
                self.idf.get(t, 0)
                * (freq_map[t] * (self.k1 + 1))
                / (freq_map[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                for t in tokens
            )
            scores.append((idx, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
