"""A minimal in-memory vector index. Swappable for FAISS/pgvector/etc. in a
real deployment — the interface (add / search) is what matters for the rest
of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data.embeddings import cosine_similarity


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict


@dataclass
class VectorStore:
    dim: int
    _vectors: np.ndarray = field(init=False)
    _texts: list[str] = field(default_factory=list, init=False)
    _metadata: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._vectors = np.zeros((0, self.dim), dtype=np.float64)

    def add(self, vectors: np.ndarray, texts: list[str], metadata: list[dict] | None = None) -> None:
        if vectors.shape[0] != len(texts):
            raise ValueError("vectors and texts must have the same length")
        metadata = metadata or [{} for _ in texts]
        self._vectors = np.vstack([self._vectors, vectors]) if self._vectors.size else vectors
        self._texts.extend(texts)
        self._metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if not self._texts:
            return []
        scores = cosine_similarity(query_vector, self._vectors)
        top_k = min(top_k, len(self._texts))
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [
            SearchResult(text=self._texts[i], score=float(scores[i]), metadata=self._metadata[i])
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._texts)
