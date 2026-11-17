from knowledge_base.service import chunk_text


def test_chunk_text_returns_no_chunks_for_empty_content():
    assert chunk_text("   ") == []


def test_chunk_text_preserves_content_across_chunks():
    source = ("Admissions information. " * 80).strip()
    chunks = chunk_text(source, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert chunks[0] in source
    assert chunks[-1][-20:] in source


def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("some text", chunk_size=200, overlap=200)
    except ValueError as exc:
        assert "Invalid" in str(exc)
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")
