"""A dependency-light embedding backend.

Rather than requiring a downloaded neural embedding model (heavy, and not
available in restricted/offline environments like CI), this implements
hashed TF-IDF: terms are hashed into a fixed-size vector space, weighted by
inverse document frequency fit on the corpus, and L2-normalized. It's a
well-understood, fast, zero-download baseline — good enough to demonstrate
(and unit test) a real chunk -> embed -> index -> search pipeline, and a
drop-in seam (`Embedder` protocol) for swapping in a real model later.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def fit(self, corpus: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashedTfidfEmbedder:
    def __init__(self, dim: int = 512, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed
        self._idf: np.ndarray | None = None

    def _hash(self, token: str) -> int:
        # Stable hash independent of PYTHONHASHSEED.
        h = 2166136261
        for ch in token:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return (h ^ self.seed) % self.dim

    def _term_freq_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        tokens = tokenize(text)
        if not tokens:
            return vec
        counts = Counter(tokens)
        for token, count in counts.items():
            vec[self._hash(token)] += count
        return vec

    def fit(self, corpus: list[str]) -> None:
        """Compute inverse-document-frequency weights from a reference corpus."""
        doc_freq = np.zeros(self.dim, dtype=np.float64)
        n_docs = max(1, len(corpus))
        for doc in corpus:
            tokens = set(tokenize(doc))
            for token in tokens:
                doc_freq[self._hash(token)] += 1
        self._idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._idf is None:
            # Fall back to unweighted term frequency if fit() wasn't called.
            idf = np.ones(self.dim, dtype=np.float64)
        else:
            idf = self._idf

        vectors = np.vstack([self._term_freq_vector(t) for t in texts]) if texts else np.zeros((0, self.dim))
        vectors = vectors * idf
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """query: (dim,), matrix: (n, dim), both assumed L2-normalized -> (n,) scores."""
    if matrix.shape[0] == 0:
        return np.zeros(0)
    return matrix @ query
