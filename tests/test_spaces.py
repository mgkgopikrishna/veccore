"""The fingerprint is the safety mechanism; these tests pin its semantics."""

import pytest

from veccore.models import EmbeddingSpace


def base(**kw):
    defaults = {"id": "s", "model_id": "all-MiniLM-L6-v2", "dimension": 384}
    defaults.update(kw)
    return EmbeddingSpace(**defaults)


def test_renaming_a_space_keeps_vectors_comparable():
    assert base(id="one").compatible_with(base(id="two"))


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "bge-small-en-v1.5"},      # different model, same dimension
        {"dimension": 768},
        {"normalize": False},
        {"pooling": "cls"},
        {"preprocessor_version": "2"},          # same model, different chunking
    ],
    ids=["model", "dimension", "normalize", "pooling", "preprocessing"],
)
def test_every_meaningful_difference_breaks_compatibility(change):
    assert not base().compatible_with(base(**change))


def test_same_dimension_different_model_is_the_dangerous_case():
    """384 == 384 means the vectors LOOK compatible. Nothing would raise. That is the bug."""
    a, b = base(model_id="all-MiniLM-L6-v2"), base(model_id="bge-small-en-v1.5")
    assert a.dimension == b.dimension
    assert not a.compatible_with(b)


def test_fingerprint_is_stable_across_instances():
    assert base().fingerprint == base().fingerprint


def test_metadata_does_not_affect_the_fingerprint():
    assert base(metadata={"note": "x"}).fingerprint == base().fingerprint


@pytest.mark.parametrize("bad", [{"dimension": 0}, {"dimension": -1}, {"model_id": ""}, {"id": ""}])
def test_invalid_spaces_are_rejected_at_construction(bad):
    with pytest.raises(ValueError):
        base(**bad)
