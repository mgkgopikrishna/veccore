"""Vectors must never leak between spaces. This is the whole premise."""

import pytest

from veccore.models import Chunk, Document, Vector
from veccore.registry import RegistryError
from veccore.stores import MemoryStore


def test_vectors_are_scoped_to_their_space():
    s = MemoryStore()
    s.put_chunks([Chunk(id="c1", doc_id="d", ordinal=0, text="t")])
    s.put_vectors([Vector(chunk_id="c1", space_id="A", values=[1.0, 0.0])])
    assert s.vector_count("A") == 1
    assert s.vector_count("B") == 0
    assert s.candidates("B") == []


def test_the_corpus_is_shared_while_vectors_are_not(core_with_candidate):
    """One copy of the text, two independent indexes over it."""
    core = core_with_candidate
    core.migrate_to("subword-v2").backfill()
    assert core.store.chunk_count() == 6
    assert core.store.vector_count("words-v1") == 6
    assert core.store.vector_count("subword-v2") == 6
    assert core.store.document_count() == 6


def test_querying_a_building_space_is_refused(core_with_candidate):
    with pytest.raises(RegistryError, match="still building"):
        core_with_candidate.query("anything", space_id="subword-v2")


def test_dropping_a_space_leaves_the_other_untouched(core_with_candidate):
    core = core_with_candidate
    core.migrate_to("subword-v2").backfill()
    removed = core.store.drop_space("subword-v2")
    assert removed == 6
    assert core.store.vector_count("words-v1") == 6
    assert core.store.chunk_count() == 6  # text survives


def test_missing_chunk_ids_drive_resumability():
    s = MemoryStore()
    s.put_chunks([Chunk(id=f"c{i}", doc_id="d", ordinal=i, text="t") for i in range(5)])
    assert len(s.chunk_ids_missing_in("A")) == 5
    s.put_vectors([Vector(chunk_id="c0", space_id="A", values=[1.0])])
    assert len(s.chunk_ids_missing_in("A")) == 4
    assert len(s.chunk_ids_missing_in("A", limit=2)) == 2


def test_an_empty_vector_is_rejected():
    with pytest.raises(ValueError):
        Vector(chunk_id="c", space_id="A", values=[])


def test_two_spaces_genuinely_disagree(core_with_candidate):
    """If the spaces ranked identically the migration machinery would be untestable."""
    core = core_with_candidate
    core.migrate_to("subword-v2").backfill()
    a = [h.chunk_id for h in core.query("container killed", k=6)]
    b = [h.chunk_id for h in core.query("container killed", k=6, space_id="subword-v2")]
    assert a != b, "the two embedding spaces produced identical rankings"


def test_document_and_chunk_hashes_track_content():
    d1, d2 = Document(id="a", text="hello"), Document(id="b", text="hello")
    assert d1.content_hash == d2.content_hash
    assert Document(id="c", text="different").content_hash != d1.content_hash
