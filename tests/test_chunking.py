"""Unit tests for chunking. Run with: pytest"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import chunk_fixed, chunk_recursive, chunk_document, Document


def test_fixed_empty():
    assert chunk_fixed("") == []


def test_fixed_overlap():
    chunks = chunk_fixed("abcdefghij", size=4, overlap=2)
    assert chunks[0] == "abcd"
    assert chunks[1] == "cdef"  # overlap of 2


def test_recursive_keeps_sentences_whole():
    text = "Première phrase. Deuxième phrase. Troisième phrase."
    chunks = chunk_recursive(text, size=30)
    joined = " ".join(chunks)
    for sent in ["Première phrase.", "Deuxième phrase.", "Troisième phrase."]:
        assert sent in joined


def test_recursive_empty():
    assert chunk_recursive("") == []


def test_chunk_document_ids():
    doc = Document(doc_id="2024-001", title="t", text="Un texte. Deux textes.")
    chunks = chunk_document(doc, strategy="recursive")
    assert all(c.doc_id == "2024-001" for c in chunks)
    assert [c.chunk_id for c in chunks] == [f"2024-001#{i}" for i in range(len(chunks))]
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_unknown_strategy_raises():
    doc = Document(doc_id="x", title="t", text="hello")
    try:
        chunk_document(doc, strategy="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
