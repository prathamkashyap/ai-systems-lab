from data.hybrid import BM25Index, reciprocal_rank_fusion
from data.pipeline import Document, RetrievalPipeline
from data.vector_store import SearchResult


def test_bm25_exact_keyword_matching():
    index = BM25Index()
    texts = [
        "Kubernetes orchestrates containerized services in production clusters.",
        "PostgreSQL is an ACID compliant relational database.",
        "PyTorch provides tensor computation and deep neural networks.",
    ]
    index.index(texts)

    results = index.search("PostgreSQL relational database", top_k=1)
    assert len(results) == 1
    assert "PostgreSQL" in results[0].text
    assert results[0].score > 0


def test_reciprocal_rank_fusion():
    list_a = [
        SearchResult(text="doc-1", score=0.9, metadata={"id": "1"}),
        SearchResult(text="doc-2", score=0.8, metadata={"id": "2"}),
    ]
    list_b = [
        SearchResult(text="doc-2", score=10.0, metadata={"id": "2"}),
        SearchResult(text="doc-3", score=5.0, metadata={"id": "3"}),
    ]

    fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_k=3)
    assert len(fused) == 3
    # doc-2 appeared high in both list_a (rank 2) and list_b (rank 1), so it should rank first in fused RRF
    assert fused[0].text == "doc-2"
    assert "rrf_score" in fused[0].metadata


def test_retrieval_pipeline_hybrid_mode():
    docs = [
        Document(id="k8s", text="Kubernetes handles horizontal pod autoscaling and deployment rollouts."),
        Document(id="redis", text="Redis is an in-memory key-value cache and data structure store."),
    ]
    pipeline = RetrievalPipeline(dim=128, chunk_size=50)
    pipeline.ingest(docs)

    results = pipeline.query("in-memory key-value cache", mode="hybrid", top_k=1)
    assert len(results) == 1
    assert results[0].metadata["source_id"] == "redis"
    assert "rrf_score" in results[0].metadata
