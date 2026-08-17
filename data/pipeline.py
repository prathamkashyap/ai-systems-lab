"""Ties chunking, embedding, BM25 indexing, and vector search into one
unified retrieval pipeline supporting dense, lexical, and hybrid (RRF) search.

    pipeline = RetrievalPipeline()
    pipeline.ingest(documents)                      # chunk -> embed -> index (vector + bm25)
    results = pipeline.query("...", mode="hybrid") # hybrid retrieval via Reciprocal Rank Fusion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from data.chunking import Chunk, chunk_by_sentences
from data.embeddings import HashedTfidfEmbedder
from data.hybrid import BM25Index, reciprocal_rank_fusion
from data.vector_store import SearchResult, VectorStore


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


class RetrievalPipeline:
    def __init__(self, dim: int = 512, chunk_size: int = 200) -> None:
        self.embedder = HashedTfidfEmbedder(dim=dim)
        self.store = VectorStore(dim=dim)
        self.bm25 = BM25Index()
        self.chunk_size = chunk_size
        self._chunks: list[Chunk] = []

    def ingest(self, documents: list[Document]) -> int:
        """Chunk every document, fit the embedder's IDF weights on the full
        chunk corpus, embed, index into both VectorStore and BM25 index.
        Returns the number of chunks indexed.
        """
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(
                chunk_by_sentences(doc.text, max_words=self.chunk_size, source_id=doc.id)
            )
        if not all_chunks:
            return 0

        texts = [c.text for c in all_chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.embed(texts)

        doc_meta_by_id = {d.id: d.metadata for d in documents}
        metadata = [
            {"source_id": c.source_id, "start_word": c.start_word, **doc_meta_by_id.get(c.source_id, {})}
            for c in all_chunks
        ]
        self.store.add(vectors, texts, metadata)
        self.bm25.index(texts, metadata)
        self._chunks.extend(all_chunks)
        return len(all_chunks)

    def query(
        self,
        text: str,
        top_k: int = 5,
        mode: Literal["vector", "bm25", "hybrid"] = "vector",
    ) -> list[SearchResult]:
        if not self._chunks:
            return []

        if mode == "bm25":
            return self.bm25.search(text, top_k=top_k)
        if mode == "vector":
            query_vector = self.embedder.embed([text])[0]
            return self.store.search(query_vector, top_k=top_k)
        if mode == "hybrid":
            query_vector = self.embedder.embed([text])[0]
            vector_results = self.store.search(query_vector, top_k=top_k * 2)
            bm25_results = self.bm25.search(text, top_k=top_k * 2)
            return reciprocal_rank_fusion([vector_results, bm25_results], top_k=top_k)

        raise ValueError(f"Unknown query mode: {mode!r}")

    def __len__(self) -> int:
        return len(self.store)
