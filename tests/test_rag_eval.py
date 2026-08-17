from ai.eval.rag_eval import (
    RAGEvaluator,
    RAGTask,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
    compute_keyword_coverage,
)


def test_context_precision_and_recall():
    retrieved = ["doc-2", "doc-1", "doc-3"]
    relevant = ["doc-1"]

    # doc-1 is at rank 2 -> MRR = 1/2 = 0.5
    assert compute_context_precision(retrieved, relevant) == 0.5
    # 1 of 1 relevant doc retrieved -> 1.0
    assert compute_context_recall(retrieved, relevant) == 1.0

    # None retrieved
    assert compute_context_precision(["doc-5"], relevant) == 0.0
    assert compute_context_recall(["doc-5"], relevant) == 0.0


def test_keyword_coverage():
    answer = "SingleFlight coalesces concurrent in-flight requests to eliminate cache stampedes."
    keywords = ["SingleFlight", "concurrent", "stampede", "missing_term"]
    score = compute_keyword_coverage(answer, keywords)
    assert score == 0.75  # 3 of 4 keywords found


def test_faithfulness_groundedness():
    context = ["PostgreSQL supports JSONB and full text search queries with GIN indexes."]
    grounded_answer = "PostgreSQL supports JSONB and full text search."
    hallucinated_answer = "Quantum entangled superconducting qubits compute exponentially."

    high_faith = compute_faithfulness(grounded_answer, context)
    low_faith = compute_faithfulness(hallucinated_answer, context)

    assert high_faith >= 0.8
    assert low_faith <= 0.2


def test_rag_evaluator_summary():
    tasks = [
        RAGTask(
            id="t1",
            question="What does SingleFlight do?",
            ground_truth="SingleFlight deduplicates in-flight requests.",
            relevant_source_ids=["doc-sf"],
            keywords=["SingleFlight", "deduplicates"],
        )
    ]
    evaluator = RAGEvaluator(tasks)
    result = evaluator.evaluate_result(
        tasks[0],
        retrieved_ids=["doc-sf"],
        contexts=["SingleFlight deduplicates concurrent in-flight requests."],
        answer="SingleFlight deduplicates in-flight requests efficiently.",
    )
    summary = evaluator.evaluate_all([result])

    assert summary.n_tasks == 1
    assert summary.mean_context_precision == 1.0
    assert summary.mean_context_recall == 1.0
    assert summary.mean_keyword_coverage == 1.0
    assert summary.mean_faithfulness > 0.5
