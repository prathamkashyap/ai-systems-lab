"""Hybrid Search: combines BM25 lexical ranking and dense vector search
using Reciprocal Rank Fusion (RRF).

Addresses the limitation of pure embedding or pure lexical search:
- Lexical search (BM25) excels at exact keywords, code tokens, IDs, and acronyms.
- Vector search excels at semantic concepts and paraphrases.
- RRF combines their ranked lists robustly without needing cross-score calibration.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from data.embeddings import tokenize
from data.vector_store import SearchResult


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: list[int] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.term_freqs: list[dict[str, int]] = []
        self.texts: list[str] = []
        self.metadata: list[dict] = []

    def index(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        self.texts = list(texts)
        self.metadata = metadata or [{} for _ in texts]
        self.corpus_size = len(texts)
        self.doc_lens = []
        self.term_freqs = []
        self.doc_freqs.clear()

        total_len = 0
        for text in texts:
            tokens = tokenize(text)
            n_tokens = len(tokens)
            self.doc_lens.append(n_tokens)
            total_len += n_tokens

            tf = Counter(tokens)
            self.term_freqs.append(dict(tf))
            for term in tf:
                self.doc_freqs[term] += 1

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size else 0.0

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self.corpus_size:
            return []

        query_tokens = tokenize(query)
        scores: list[float] = [0.0] * self.corpus_size

        for term in query_tokens:
            df = self.doc_freqs.get(term, 0)
            if not df:
                continue
            # Standard Lucene/BM25 IDF
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
            for doc_idx, tf_dict in enumerate(self.term_freqs):
                tf = tf_dict.get(term, 0)
                if not tf:
                    continue
                doc_len = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                scores[doc_idx] += idf * (numerator / denominator)

        scored_indices = [i for i in range(self.corpus_size) if scores[i] > 0]
        scored_indices.sort(key=lambda i: scores[i], reverse=True)
        top_indices = scored_indices[:top_k]

        return [
            SearchResult(text=self.texts[i], score=scores[i], metadata=self.metadata[i])
            for i in top_indices
        ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
    top_k: int = 5,
) -> list[SearchResult]:
    """Combines multiple ranked lists into a single fused ranking via RRF.

    score(d) = sum_{m in rankings} 1 / (k + rank_m(d))
    """
    fused_scores: dict[str, float] = defaultdict(float)
    doc_registry: dict[str, SearchResult] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            key = result.text
            doc_registry[key] = result
            fused_scores[key] += 1.0 / (k + rank)

    sorted_docs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    results: list[SearchResult] = []
    for text, score in sorted_docs[:top_k]:
        orig = doc_registry[text]
        fused_meta = {**orig.metadata, "rrf_score": score}
        results.append(SearchResult(text=text, score=score, metadata=fused_meta))

    return results
