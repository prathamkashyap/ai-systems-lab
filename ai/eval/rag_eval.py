"""RAG (Retrieval-Augmented Generation) Evaluation Suite.

Provides standardized evaluation metrics for retrieval and generation quality:
- Context Precision: Measures if retrieved context ranks relevant documents first (Mean Reciprocal Rank).
- Context Recall: Measures what fraction of ground truth relevant documents were retrieved.
- Keyword Coverage: Checks whether the expected factual key terms are present in the generation.
- Faithfulness / Groundedness: Measures whether the generated answer claims are supported by
  the retrieved context (hallucination reduction).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from data.embeddings import tokenize


@dataclass
class RAGTask:
    id: str
    question: str
    ground_truth: str = ""
    relevant_source_ids: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class RAGResult:
    task_id: str
    question: str
    retrieved_source_ids: list[str]
    retrieved_contexts: list[str]
    generated_answer: str

    context_precision: float = 0.0
    context_recall: float = 0.0
    keyword_coverage: float = 0.0
    faithfulness: float = 0.0


@dataclass
class RAGEvalSummary:
    n_tasks: int
    mean_context_precision: float
    mean_context_recall: float
    mean_keyword_coverage: float
    mean_faithfulness: float
    results: list[RAGResult]

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "mean_context_precision": round(self.mean_context_precision, 4),
            "mean_context_recall": round(self.mean_context_recall, 4),
            "mean_keyword_coverage": round(self.mean_keyword_coverage, 4),
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "results": [r.__dict__ for r in self.results],
        }


def compute_context_precision(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank (MRR) / Precision score for retrieved source IDs."""
    if not relevant_ids:
        return 1.0
    if not retrieved_ids:
        return 0.0

    target_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in target_set:
            return 1.0 / rank
    return 0.0


def compute_context_recall(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Fraction of relevant sources retrieved."""
    if not relevant_ids:
        return 1.0
    if not retrieved_ids:
        return 0.0

    target_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    hits = sum(1 for rid in target_set if rid in retrieved_set)
    return hits / len(target_set)


def compute_keyword_coverage(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the generated answer."""
    if not keywords:
        return 1.0
    lowered = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lowered)
    return hits / len(keywords)


def compute_faithfulness(answer: str, retrieved_contexts: list[str]) -> float:
    """Estimates lexical groundedness of the generated answer within the retrieved contexts.

    Computes token overlap between content tokens of the answer and the context text.
    """
    if not answer.strip():
        return 0.0
    if not retrieved_contexts:
        return 0.0

    context_tokens = set()
    for ctx in retrieved_contexts:
        context_tokens.update(tokenize(ctx))

    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0

    # Filter trivial stop words for groundedness check
    stop_words = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or", "is",
        "are", "was", "were", "it", "this", "that", "with", "by", "as", "from",
    }
    content_tokens = [t for t in answer_tokens if t not in stop_words and len(t) > 2]
    if not content_tokens:
        return 1.0

    hits = sum(1 for t in content_tokens if t in context_tokens)
    return hits / len(content_tokens)


class RAGEvaluator:
    """Evaluates RAG outputs against defined RAG benchmark tasks."""

    def __init__(self, tasks: list[RAGTask]) -> None:
        self.tasks = tasks

    def evaluate_result(self, task: RAGTask, retrieved_ids: list[str], contexts: list[str], answer: str) -> RAGResult:
        prec = compute_context_precision(retrieved_ids, task.relevant_source_ids)
        rec = compute_context_recall(retrieved_ids, task.relevant_source_ids)
        kw = compute_keyword_coverage(answer, task.keywords)
        faith = compute_faithfulness(answer, contexts)

        return RAGResult(
            task_id=task.id,
            question=task.question,
            retrieved_source_ids=retrieved_ids,
            retrieved_contexts=contexts,
            generated_answer=answer,
            context_precision=prec,
            context_recall=rec,
            keyword_coverage=kw,
            faithfulness=faith,
        )

    def evaluate_all(self, results: list[RAGResult]) -> RAGEvalSummary:
        if not results:
            return RAGEvalSummary(0, 0.0, 0.0, 0.0, 0.0, [])

        return RAGEvalSummary(
            n_tasks=len(results),
            mean_context_precision=sum(r.context_precision for r in results) / len(results),
            mean_context_recall=sum(r.context_recall for r in results) / len(results),
            mean_keyword_coverage=sum(r.keyword_coverage for r in results) / len(results),
            mean_faithfulness=sum(r.faithfulness for r in results) / len(results),
            results=results,
        )
