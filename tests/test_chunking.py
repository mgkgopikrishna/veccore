import pytest

from veccore.chunking import chunk_document
from veccore.models import Document


def test_short_document_is_one_chunk():
    chunks = chunk_document(Document(id="d", text="One short sentence."))
    assert len(chunks) == 1
    assert chunks[0].id == "d::0"
    assert chunks[0].doc_id == "d"


def test_empty_document_yields_nothing():
    assert chunk_document(Document(id="d", text="   ")) == []


def test_long_document_splits_and_stays_within_budget():
    text = " ".join(f"Sentence number {i} with filler words." for i in range(80))
    chunks = chunk_document(Document(id="d", text=text), max_chars=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_ordinals_are_sequential():
    text = " ".join(f"Sentence {i}." for i in range(50))
    chunks = chunk_document(Document(id="d", text=text), max_chars=100, overlap=20)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_a_sentence_longer_than_the_budget_is_not_dropped():
    giant = "x" * 900
    chunks = chunk_document(Document(id="d", text=giant), max_chars=200, overlap=20)
    assert chunks
    assert sum(len(c.text) for c in chunks) >= 900 - 50


def test_metadata_propagates_to_chunks():
    doc = Document(id="d", text="Body text here.", metadata={"source": "runbook"})
    assert chunk_document(doc)[0].metadata == {"source": "runbook"}


@pytest.mark.parametrize("kw", [{"max_chars": 0}, {"max_chars": -5}, {"overlap": 800}])
def test_invalid_parameters_are_rejected(kw):
    with pytest.raises(ValueError):
        chunk_document(Document(id="d", text="text"), **{"max_chars": 800, **kw})
