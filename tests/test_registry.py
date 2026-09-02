"""The one-active-space invariant, and every transition that must be refused."""

import pytest

from tests.helpers import space_v1, space_v2
from veccore.models import SpaceState
from veccore.registry import RegistryError, SpaceRegistry


@pytest.fixture
def reg():
    r = SpaceRegistry()
    r.register(space_v1().space, activate=True)
    r.register(space_v2().space)
    return r


def test_exactly_one_space_is_active(reg):
    assert reg.active.space.id == "words-v1"
    assert sum(s.state is SpaceState.ACTIVE for s in reg.all()) == 1


def test_a_second_space_cannot_be_activated_directly(reg):
    with pytest.raises(RegistryError, match="already active"):
        reg.register(space_v2("third").space, activate=True)


def test_incomplete_backfill_blocks_promotion(reg):
    with pytest.raises(RegistryError, match="un-embedded"):
        reg.promote_to_shadow("subword-v2", missing=3)
    assert reg.get("subword-v2").state is SpaceState.BUILDING


def test_cutover_requires_shadow_state(reg):
    with pytest.raises(RegistryError, match="not shadow"):
        reg.cutover("subword-v2")


def test_a_refused_cutover_changes_nothing(reg):
    before = {s.space.id: s.state for s in reg.all()}
    with pytest.raises(RegistryError):
        reg.cutover("subword-v2")
    assert {s.space.id: s.state for s in reg.all()} == before


def test_cutover_retires_the_outgoing_space_rather_than_deleting_it(reg):
    reg.promote_to_shadow("subword-v2", missing=0)
    new, old = reg.cutover("subword-v2")
    assert new.state is SpaceState.ACTIVE
    assert old.state is SpaceState.RETIRED
    assert reg.get("words-v1") is old  # still registered, so rollback is possible


def test_rollback_restores_the_retired_space(reg):
    reg.promote_to_shadow("subword-v2", missing=0)
    reg.cutover("subword-v2")
    restored, demoted = reg.rollback("words-v1")
    assert restored.state is SpaceState.ACTIVE
    assert demoted.state is SpaceState.SHADOW
    assert reg.active.space.id == "words-v1"


def test_cannot_roll_back_to_a_space_that_was_never_retired(reg):
    with pytest.raises(RegistryError, match="not retired"):
        reg.rollback("subword-v2")


def test_retiring_the_active_space_is_refused(reg):
    with pytest.raises(RegistryError, match="while it is active"):
        reg.retire("words-v1")


def test_only_retired_spaces_can_be_forgotten(reg):
    with pytest.raises(RegistryError):
        reg.forget("words-v1")


def test_unknown_space_is_a_clear_error(reg):
    with pytest.raises(RegistryError, match="unknown space"):
        reg.get("nope")


def test_registry_with_no_active_space_refuses_to_serve():
    with pytest.raises(RegistryError, match="no active embedding space"):
        SpaceRegistry().require_active()
