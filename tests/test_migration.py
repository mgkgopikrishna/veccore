"""The migration lifecycle: backfill, compare, cutover, rollback."""

import pytest

from tests.helpers import QUERIES, space_v1
from veccore.models import SpaceState
from veccore.registry import RegistryError


# --- backfill ------------------------------------------------------------------- #
def test_backfill_embeds_the_existing_corpus_into_the_new_space(core_with_candidate):
    core = core_with_candidate
    assert core.store.vector_count("subword-v2") == 0
    progress = core.migrate_to("subword-v2").backfill()
    assert progress.embedded == 6
    assert progress.complete
    assert core.store.vector_count("subword-v2") == 6


def test_backfill_is_resumable(core_with_candidate):
    """Stop after one small batch, then continue. Nothing is re-embedded."""
    core = core_with_candidate
    m = core.migrate_to("subword-v2")

    first = m.backfill(batch_size=2, max_batches=1)
    assert first.embedded == 2
    assert first.remaining == 4
    assert not first.complete

    second = m.backfill(batch_size=2)
    assert second.embedded == 4          # only the four that were missing
    assert second.complete
    assert core.store.vector_count("subword-v2") == 6


def test_an_incomplete_backfill_leaves_the_space_building(core_with_candidate):
    core = core_with_candidate
    core.migrate_to("subword-v2").backfill(batch_size=1, max_batches=1)
    assert core.registry.get("subword-v2").state is SpaceState.BUILDING


def test_completed_backfill_promotes_to_shadow(core_with_candidate):
    core = core_with_candidate
    core.migrate_to("subword-v2").backfill()
    assert core.registry.get("subword-v2").state is SpaceState.SHADOW


def test_the_live_space_keeps_serving_throughout(core_with_candidate):
    core = core_with_candidate
    m = core.migrate_to("subword-v2")
    m.backfill(batch_size=1, max_batches=2)
    hits = core.query("terraform lock stuck", k=3)   # still served by v1
    assert hits
    assert core.registry.active.space.id == "words-v1"


def test_migrating_into_an_identical_space_is_rejected(core):
    core.add_space(space_v1("words-v1-copy"))
    with pytest.raises(RegistryError, match="nothing to"):
        core.migrate_to("words-v1-copy")


# --- dual write ------------------------------------------------------------------ #
def test_new_documents_land_in_both_spaces(core_with_candidate):
    from veccore.models import Document

    core = core_with_candidate
    core.migrate_to("subword-v2").backfill()
    core.ingest(Document(id="new", text="A sidecar container shares the pod network namespace."))
    assert core.store.vector_count("words-v1") == 7
    assert core.store.vector_count("subword-v2") == 7


def test_retired_spaces_stop_receiving_writes(core_with_candidate):
    from veccore.models import Document

    core = core_with_candidate
    m = core.migrate_to("subword-v2")
    m.backfill()
    m.cutover()
    before = core.store.vector_count("words-v1")
    core.ingest(Document(id="later", text="Node taints repel pods that lack a toleration."))
    assert core.store.vector_count("words-v1") == before      # retired: frozen
    assert core.store.vector_count("subword-v2") == before + 1


# --- shadow comparison ------------------------------------------------------------ #
def test_comparison_reports_real_divergence(core_with_candidate):
    core = core_with_candidate
    m = core.migrate_to("subword-v2")
    m.backfill()
    report = m.compare(QUERIES, k=3)

    assert report.queries == len(QUERIES)
    assert 0.0 <= report.mean_overlap <= 1.0
    assert 0.0 <= report.top1_agreement <= 1.0
    assert report.mean_rank_shift >= 0.0
    assert len(report.per_query) == len(QUERIES)
    assert report.verdict


def test_comparing_an_incomplete_space_is_refused(core_with_candidate):
    core = core_with_candidate
    m = core.migrate_to("subword-v2")
    m.backfill(batch_size=1, max_batches=1)
    with pytest.raises(RegistryError, match="still building"):
        m.compare(QUERIES)


def test_comparing_a_space_against_itself_would_be_meaningless(core):
    """Guarded at construction: identical fingerprints cannot form a Migration."""
    core.add_space(space_v1("clone"))
    with pytest.raises(RegistryError):
        core.migrate_to("clone")


def test_empty_query_list_yields_an_empty_report(core_with_candidate):
    core = core_with_candidate
    m = core.migrate_to("subword-v2")
    m.backfill()
    assert m.compare([], k=3).queries == 0


# --- cutover and rollback --------------------------------------------------------- #
def test_cutover_swaps_active_and_retires_the_old(core_with_candidate):
    core = core_with_candidate
    m = core.migrate_to("subword-v2")
    m.backfill()
    new, old = m.cutover()
    assert new.space.id == "subword-v2"
    assert old.space.id == "words-v1"
    assert core.registry.active.space.id == "subword-v2"
    assert core.registry.get("words-v1").state is SpaceState.RETIRED


def test_cutover_can_be_gated_on_measured_overlap(core_with_candidate):
    core = core_with_candidate
    m = core.migrate_to("subword-v2")
    m.backfill()
    report = m.compare(QUERIES, k=3)

    with pytest.raises(RegistryError, match="below the required"):
        m.cutover(min_overlap=1.01, report=report)     # impossible bar
    assert core.registry.active.space.id == "words-v1"  # unchanged

    m.cutover(min_overlap=0.0, report=report)
    assert core.registry.active.space.id == "subword-v2"


def test_gating_without_a_report_is_an_error(core_with_candidate):
    core = core_with_candidate
    m = core.migrate_to("subword-v2")
    m.backfill()
    with pytest.raises(RegistryError, match="requires a DivergenceReport"):
        m.cutover(min_overlap=0.8)


def test_cutover_before_backfill_is_refused(core_with_candidate):
    core = core_with_candidate
    with pytest.raises(RegistryError, match="not shadow"):
        core.migrate_to("subword-v2").cutover()


def test_rollback_restores_the_previous_space_without_reindexing(core_with_candidate):
    core = core_with_candidate
    m = core.migrate_to("subword-v2")
    m.backfill()
    m.cutover()
    vectors_before = core.store.vector_count("words-v1")

    restored, demoted = core.rollback_to("words-v1")

    assert restored.space.id == "words-v1"
    assert demoted.space.id == "subword-v2"
    assert core.registry.active.space.id == "words-v1"
    assert core.store.vector_count("words-v1") == vectors_before  # never deleted


def test_queries_follow_the_active_space_across_cutover(core_with_candidate):
    core = core_with_candidate
    m = core.migrate_to("subword-v2")
    m.backfill()

    before = [h.chunk_id for h in core.query("why was my container killed", k=6)]
    m.cutover()
    after = [h.chunk_id for h in core.query("why was my container killed", k=6)]

    assert before != after, "cutover did not change which space serves reads"
