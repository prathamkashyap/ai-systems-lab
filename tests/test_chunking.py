from data.chunking import chunk_by_sentences, chunk_by_words


def test_chunk_by_words_basic():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_by_words(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    # every chunk except possibly the last should be exactly chunk_size words
    for c in chunks[:-1]:
        assert len(c.text.split()) == 100


def test_chunk_by_words_overlap_continuity():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_by_words(text, chunk_size=100, overlap=20)
    first_tail = chunks[0].text.split()[-20:]
    second_head = chunks[1].text.split()[:20]
    assert first_tail == second_head


def test_chunk_by_words_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_by_words("some text here", chunk_size=10, overlap=10)


def test_chunk_by_words_empty_text():
    assert chunk_by_words("") == []


def test_chunk_by_sentences_respects_max_words():
    text = "This is sentence one. This is sentence two. This is sentence three. " * 20
    chunks = chunk_by_sentences(text, max_words=30)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= 60  # allow slack for overlap sentence(s)


def test_chunk_by_sentences_source_id_propagates():
    chunks = chunk_by_sentences("One. Two. Three.", max_words=50, source_id="doc-42")
    assert all(c.source_id == "doc-42" for c in chunks)
