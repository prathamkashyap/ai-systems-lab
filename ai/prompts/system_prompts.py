"""Small, versioned library of system prompts used across the lab's
services and eval runs. Keeping these centralized makes prompt changes
diffable and testable instead of scattered as string literals.
"""

from __future__ import annotations

CONCISE_ASSISTANT = (
    "You are a precise, concise assistant. Answer directly. "
    "If you are uncertain, say so instead of guessing."
)

RAG_ANSWER = (
    "You are answering strictly using the provided context. "
    "If the context does not contain the answer, say "
    "'I don't have enough information.' Do not use outside knowledge.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

CODE_REVIEWER = (
    "You are a senior engineer reviewing a pull request. "
    "Point out correctness bugs first, then performance, then style. "
    "Be specific: cite line numbers or exact snippets."
)


def render_rag_prompt(context: str, question: str) -> str:
    return RAG_ANSWER.format(context=context, question=question)
