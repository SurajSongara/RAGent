"""What to do after a stage succeeds or fails, decided without touching IO.

Keeping this pure is what makes the awkward cases testable: retry exhaustion,
poison messages, a document that fans out to two stages and back in to one, and
resuming a run that died halfway. The consumer in `ragent.workers.run` supplies
the broker and the database; everything it decides comes from here.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragent.ingest.formats import FormatFamily, UnsupportedFormatError
from ragent.pipeline.messages import MalformedMessageError, StageMessage
from ragent.pipeline.stages import (
    STAGES_BY_NAME,
    TERMINAL_STAGE,
    Stage,
    is_complete,
    ready_stages,
    retry_delay_ms,
)
from ragent.pipeline.topology import retry_exchange_name

__all__ = [
    "PermanentError",
    "Retry",
    "DeadLetter",
    "plan_failure",
    "next_messages",
    "document_ready",
    "resume_plan",
]


class PermanentError(Exception):
    """A failure that retrying cannot fix.

    An encrypted PDF is still encrypted in five minutes. Burning three attempts
    and two backoff tiers to rediscover that just delays the operator seeing it
    in the DLQ.
    """


#: Failures that are permanent by nature, wherever they are raised.
_PERMANENT: tuple[type[BaseException], ...] = (
    PermanentError,
    UnsupportedFormatError,
    MalformedMessageError,
)


@dataclass(frozen=True, slots=True)
class Retry:
    delay_ms: int
    exchange: str
    attempt: int


@dataclass(frozen=True, slots=True)
class DeadLetter:
    reason: str


def plan_failure(message: StageMessage, exc: BaseException) -> Retry | DeadLetter:
    """Decide between a delayed retry and the dead-letter queue."""
    stage = STAGES_BY_NAME.get(message.stage)
    if stage is None:
        return DeadLetter(f"unknown stage {message.stage!r}")

    if isinstance(exc, _PERMANENT):
        return DeadLetter(f"{type(exc).__name__}: {exc}")

    delay = retry_delay_ms(stage, message.attempt)
    if delay is None:
        return DeadLetter(
            f"exhausted {stage.max_attempts} attempts; last error {type(exc).__name__}: {exc}"
        )
    return Retry(delay_ms=delay, exchange=retry_exchange_name(delay), attempt=message.attempt + 1)


def next_messages(
    message: StageMessage,
    completed: frozenset[str] | set[str],
    dispatched: frozenset[str] | set[str] = frozenset(),
) -> tuple[StageMessage, ...]:
    """Messages to publish after `message.stage` succeeded.

    `completed` and `dispatched` come from the document's `ingest_stages` rows,
    read in the same transaction that marked this stage succeeded. That read is
    what makes the handoff safe when two stages finish at once.
    """
    return tuple(
        message.for_stage(stage.name)
        for stage in ready_stages(message.family, completed, dispatched)
    )


def document_ready(family: FormatFamily, completed: frozenset[str] | set[str]) -> bool:
    """True once every stage on this family's path has succeeded."""
    return is_complete(family, completed)


def resume_plan(
    family: FormatFamily,
    completed: frozenset[str] | set[str],
    dispatched: frozenset[str] | set[str] = frozenset(),
) -> tuple[Stage, ...]:
    """Stages to re-queue for a document whose run was interrupted.

    Called at worker startup for anything left `processing`. A stage that was
    mid-flight when the worker died has a row but never succeeded, so it is
    passed as dispatched-but-not-completed and is deliberately re-queued: stages
    are idempotent, and re-running one is cheaper than stalling the document.
    """
    completed = frozenset(completed)
    in_flight = frozenset(dispatched) - completed
    ready = ready_stages(family, completed, dispatched)
    return tuple(STAGES_BY_NAME[name] for name in sorted(in_flight)) + ready


def terminal_stage_name() -> str:
    return TERMINAL_STAGE
