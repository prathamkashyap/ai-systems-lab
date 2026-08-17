"""Text chunking for retrieval pipelines.

Implements fixed-size word chunking with overlap, plus a sentence-aware
variant that avoids splitting mid-sentence where possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    start_word: int
    end_word: int
    source_id: str = ""


def chunk_by_words(text: str, chunk_size: int = 200, overlap: int = 40, source_id: str = "") -> list[Chunk]:
    """Split `text` into overlapping windows of `chunk_size` words.

    `overlap` words are repeated between consecutive chunks so context isn't
    lost at chunk boundaries — a common retrieval-quality lever.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(
            Chunk(text=" ".join(chunk_words), start_word=start, end_word=end, source_id=source_id)
        )
        if end == len(words):
            break
    return chunks


def chunk_by_sentences(
    text: str, max_words: int = 200, overlap_sentences: int = 1, source_id: str = ""
) -> list[Chunk]:
    """Group whole sentences into chunks up to `max_words`, keeping the last
    `overlap_sentences` of one chunk as the start of the next for continuity.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0
    word_cursor = 0
    chunk_start = 0

    def flush() -> None:
        nonlocal current, current_words, chunk_start
        if current:
            chunks.append(
                Chunk(
                    text=" ".join(current),
                    start_word=chunk_start,
                    end_word=word_cursor,
                    source_id=source_id,
                )
            )

    for sentence in sentences:
        n_words = len(sentence.split())
        if current and current_words + n_words > max_words:
            flush()
            keep = current[-overlap_sentences:] if overlap_sentences else []
            chunk_start = word_cursor - sum(len(s.split()) for s in keep)
            current = list(keep)
            current_words = sum(len(s.split()) for s in keep)
        current.append(sentence)
        current_words += n_words
        word_cursor += n_words

    flush()
    return chunks
