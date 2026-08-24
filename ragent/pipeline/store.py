"""Where DAG state lives between stages.

Backed by `ingest_runs` / `ingest_stages` in production. The protocol exists so
the scheduling logic can be exercised without a database, and so the consumer
never reaches for a session directly.

The one rule that matters: `complete_stage` must record the completion and read
back the run's state atomically. Two stages finishing at the same instant is the
normal case here — `tables` and `figures` run concurrently and both feed
`chunk` — and a read-then-write race would either publish `chunk` twice or not
at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = ["StageSnapshot", "StageStore", "InMemoryStageStore"]


@dataclass(frozen=True, slots=True)
class StageSnapshot:
    """A run's progress at one moment."""

    #: Stages that succeeded.
    completed: frozenset[str]
    #: Stages with a row in any status — queued, running, succeeded or failed.
    #: Feeds `ready_stages(dispatched=...)` to prevent double-publishing.
    dispatched: frozenset[str]

    @property
    def in_flight(self) -> frozenset[str]:
        return self.dispatched - self.completed


class StageStore(Protocol):
    async def mark_dispatched(self, run_id: str, stages: tuple[str, ...]) -> None:
        """Record that stages have been queued, before they are published."""
        ...

    async def begin_stage(self, run_id: str, stage: str, attempt: int) -> None: ...

    async def complete_stage(
        self, run_id: str, stage: str, metrics: dict[str, Any] | None = None
    ) -> StageSnapshot:
        """Mark succeeded and return the run's state, atomically."""
        ...

    async def fail_stage(self, run_id: str, stage: str, error: str, *, permanent: bool) -> None: ...

    async def snapshot(self, run_id: str) -> StageSnapshot: ...


@dataclass
class InMemoryStageStore:
    """Reference implementation. Used by tests and by `make bench` dry runs."""

    _completed: dict[str, set[str]] = field(default_factory=dict)
    _dispatched: dict[str, set[str]] = field(default_factory=dict)
    _errors: dict[str, dict[str, str]] = field(default_factory=dict)
    _attempts: dict[str, dict[str, int]] = field(default_factory=dict)

    async def mark_dispatched(self, run_id: str, stages: tuple[str, ...]) -> None:
        self._dispatched.setdefault(run_id, set()).update(stages)

    async def begin_stage(self, run_id: str, stage: str, attempt: int) -> None:
        self._dispatched.setdefault(run_id, set()).add(stage)
        self._attempts.setdefault(run_id, {})[stage] = attempt

    async def complete_stage(
        self, run_id: str, stage: str, metrics: dict[str, Any] | None = None
    ) -> StageSnapshot:
        self._completed.setdefault(run_id, set()).add(stage)
        self._dispatched.setdefault(run_id, set()).add(stage)
        return await self.snapshot(run_id)

    async def fail_stage(self, run_id: str, stage: str, error: str, *, permanent: bool) -> None:
        self._errors.setdefault(run_id, {})[stage] = error
        if permanent:
            # Leave it dispatched so the scheduler does not immediately retry it;
            # a permanent failure needs a human, not another attempt.
            self._dispatched.setdefault(run_id, set()).add(stage)

    async def snapshot(self, run_id: str) -> StageSnapshot:
        return StageSnapshot(
            completed=frozenset(self._completed.get(run_id, set())),
            dispatched=frozenset(self._dispatched.get(run_id, set())),
        )

    def errors(self, run_id: str) -> dict[str, str]:
        return dict(self._errors.get(run_id, {}))
