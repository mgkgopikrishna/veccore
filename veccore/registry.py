"""Space registry: holds the one invariant everything else depends on.

    At most one space is ACTIVE at any moment, and only an ACTIVE space serves reads.

That single rule is what makes a migration safe. A half-backfilled space cannot be
queried, because it is BUILDING. A fully-backfilled candidate can be compared against
production, because it is SHADOW. Promotion is one atomic transition, and the space it
replaces is RETIRED rather than deleted -- so rollback is a state change, not a rebuild.

Transitions are validated before anything mutates. A rejected transition leaves the
registry exactly as it was.
"""

from __future__ import annotations

from .models import EmbeddingSpace, SpaceState, SpaceStatus


class RegistryError(RuntimeError):
    pass


class SpaceRegistry:
    def __init__(self) -> None:
        self._spaces: dict[str, SpaceStatus] = {}

    # --- registration ----------------------------------------------------------- #
    def register(self, space: EmbeddingSpace, *, activate: bool = False) -> SpaceStatus:
        if space.id in self._spaces:
            raise RegistryError(f"space {space.id!r} is already registered")
        if activate and self.active is not None:
            raise RegistryError(
                f"cannot activate {space.id!r} directly: {self.active.space.id!r} is "
                "already active. Run a migration and cut over instead."
            )
        state = SpaceState.ACTIVE if activate else SpaceState.BUILDING
        status = SpaceStatus(space=space, state=state)
        self._spaces[space.id] = status
        return status

    def get(self, space_id: str) -> SpaceStatus:
        try:
            return self._spaces[space_id]
        except KeyError:
            raise RegistryError(f"unknown space {space_id!r}") from None

    def all(self) -> list[SpaceStatus]:
        return list(self._spaces.values())

    # --- the invariant ----------------------------------------------------------- #
    @property
    def active(self) -> SpaceStatus | None:
        found = [s for s in self._spaces.values() if s.state is SpaceState.ACTIVE]
        if len(found) > 1:  # pragma: no cover - defensive; transitions prevent this
            raise RegistryError(
                f"registry invariant violated: {len(found)} active spaces "
                f"({[s.space.id for s in found]})"
            )
        return found[0] if found else None

    @property
    def shadow(self) -> SpaceStatus | None:
        found = [s for s in self._spaces.values() if s.state is SpaceState.SHADOW]
        return found[0] if found else None

    def require_active(self) -> SpaceStatus:
        a = self.active
        if a is None:
            raise RegistryError(
                "no active embedding space. Register one with activate=True before "
                "ingesting or querying."
            )
        return a

    # --- transitions -------------------------------------------------------------- #
    def promote_to_shadow(self, space_id: str, *, missing: int) -> SpaceStatus:
        """BUILDING -> SHADOW. Refused while the backfill is incomplete."""
        status = self.get(space_id)
        if status.state not in (SpaceState.BUILDING, SpaceState.SHADOW):
            raise RegistryError(
                f"cannot promote {space_id!r} from {status.state.value} to shadow"
            )
        if missing > 0:
            raise RegistryError(
                f"cannot promote {space_id!r} to shadow: {missing} chunk(s) still "
                "un-embedded. A partially backfilled space would return incomplete "
                "results, so it is not allowed to serve or be compared."
            )
        status.state = SpaceState.SHADOW
        return status

    def cutover(self, space_id: str) -> tuple[SpaceStatus, SpaceStatus | None]:
        """SHADOW -> ACTIVE, and the previous ACTIVE -> RETIRED. Atomic.

        Everything that can fail is checked before the first mutation, so a rejected
        cutover leaves both spaces exactly where they were.
        """
        incoming = self.get(space_id)
        if incoming.state is not SpaceState.SHADOW:
            raise RegistryError(
                f"cannot cut over to {space_id!r}: it is {incoming.state.value}, not "
                "shadow. Backfill it and promote it first."
            )
        outgoing = self.active
        if outgoing is not None and outgoing.space.id == space_id:  # pragma: no cover
            raise RegistryError(f"{space_id!r} is already active")

        # --- no failures possible past this line ---
        if outgoing is not None:
            outgoing.state = SpaceState.RETIRED
        incoming.state = SpaceState.ACTIVE
        return incoming, outgoing

    def rollback(self, to_space_id: str) -> tuple[SpaceStatus, SpaceStatus]:
        """Put a RETIRED space back in front. The reason retirement is not deletion."""
        target = self.get(to_space_id)
        if target.state is not SpaceState.RETIRED:
            raise RegistryError(
                f"cannot roll back to {to_space_id!r}: it is {target.state.value}, "
                "not retired"
            )
        current = self.require_active()

        current.state = SpaceState.SHADOW
        target.state = SpaceState.ACTIVE
        return target, current

    def retire(self, space_id: str) -> SpaceStatus:
        status = self.get(space_id)
        if status.state is SpaceState.ACTIVE:
            raise RegistryError(
                f"refusing to retire {space_id!r} while it is active -- that would "
                "leave the system with nothing serving reads"
            )
        status.state = SpaceState.RETIRED
        return status

    def forget(self, space_id: str) -> None:
        status = self.get(space_id)
        if status.state is not SpaceState.RETIRED:
            raise RegistryError(f"only retired spaces can be removed; {space_id!r} is {status.state.value}")
        del self._spaces[space_id]
