from data.embeddings import HashedTfidfEmbedder
from data.pipeline import Document, RetrievalPipeline
from data.vector_store import VectorStore


def test_vector_store_search_returns_best_match_first():
    embedder = HashedTfidfEmbedder(dim=64)
    texts = [
        "The cat sat on the mat.",
        "Quantum computers use qubits for computation.",
        "Dogs are loyal animals that bark.",
    ]
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    store = VectorStore(dim=64)
    store.add(vectors, texts)

    query_vec = embedder.embed(["Tell me about qubits and quantum computation"])[0]
    results = store.search(query_vec, top_k=2)

    assert len(results) == 2
    assert "Quantum" in results[0].text


def test_vector_store_len():
    embedder = HashedTfidfEmbedder(dim=32)
    texts = ["a b c", "d e f"]
    embedder.fit(texts)
    store = VectorStore(dim=32)
    store.add(embedder.embed(texts), texts)
    assert len(store) == 2


def test_retrieval_pipeline_end_to_end():
    docs = [
        Document(id="doc1", text="Python is a dynamically typed programming language. " * 5),
        Document(id="doc2", text="The Eiffel Tower is located in Paris, France. " * 5),
    ]
    pipeline = RetrievalPipeline(dim=128, chunk_size=50)
    n_chunks = pipeline.ingest(docs)
    assert n_chunks > 0
    assert len(pipeline) == n_chunks

    results = pipeline.query("Where is the Eiffel Tower?", top_k=1)
    assert len(results) == 1
    assert results[0].metadata["source_id"] == "doc2"


def test_retrieval_pipeline_empty_ingest():
    pipeline = RetrievalPipeline()
    assert pipeline.ingest([]) == 0
    assert pipeline.query("anything") == []
