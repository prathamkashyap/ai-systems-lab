# ADR 0005: Hybrid retrieval with Reciprocal Rank Fusion (RRF)

## Status
Accepted

## Context
Pure vector search (dense embeddings) excels at semantic understanding but struggles with exact keyword
matches, domain acronyms, variable names, and error codes. Conversely, pure BM25 lexical search misses
paraphrases and semantic synonyms. A production RAG pipeline requires combining both lexical and dense
approaches. Combining raw similarity scores from different algorithms is fragile due to disparate score scales.

## Decision
Implement `BM25Index` and Reciprocal Rank Fusion (`reciprocal_rank_fusion` in `data/hybrid.py`):
- Compute dense top-K results via cosine similarity on the vector index.
- Compute sparse top-K results via BM25 lexical term frequency / inverse document frequency.
- Fuse results using standard RRF:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{vector}, \text{bm25}\}} \frac{1}{k + \text{rank}_m(d)}$$
  where $k = 60$.
- Unified `RetrievalPipeline.query(text, mode="hybrid" | "vector" | "bm25")` interface.

## Consequences
- **Positive:** Significant improvement in retrieval recall and precision across both keyword queries and natural language questions.
- **Positive:** Zero external dependencies (no Elasticsearch / Java / external search engine required).
- **Positive:** RRF score calibration is parameter-free and resistant to outlier vector distance scores.
