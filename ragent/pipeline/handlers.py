"""Stage handler registry.

Handlers register by stage name and receive only a `StageMessage`; anything they
need they read from Postgres or MinIO themselves. That keeps the consumer free
of stage-specific wiring and keeps each stage independently testable.

A stage with no registered handler is a *permanent* failure, not a retry. The
handler is not going to appear five minutes later, and the DLQ entry names the
stage, which is a far better signal than a queue quietly filling up.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ragent.pipeline.messages import StageMessage
from ragent.pipeline.stages import STAGES_BY_NAME

__all__ = ["StageHandler", "HandlerNotRegistered", "handler", "get_handler", "registered_stages"]

#: Returns metrics recorded against `ingest_stages.metrics`.
StageHandler = Callable[[StageMessage], Awaitable[dict[str, Any] | None]]

_HANDLERS: dict[str, StageHandler] = {}


class HandlerNotRegistered(LookupError):
    pass


def handler(stage_name: str) -> Callable[[StageHandler], StageHandler]:
    """Register the handler for a stage. Rejects typos at import time."""
    if stage_name not in STAGES_BY_NAME:
        raise ValueError(f"unknown stage {stage_name!r}; expected one of {sorted(STAGES_BY_NAME)}")

    def register(fn: StageHandler) -> StageHandler:
        if stage_name in _HANDLERS:
            raise ValueError(f"stage {stage_name!r} already has a handler")
        _HANDLERS[stage_name] = fn
        return fn

    return register


def get_handler(stage_name: str) -> StageHandler:
    try:
        return _HANDLERS[stage_name]
    except KeyError:
        raise HandlerNotRegistered(f"no handler registered for stage {stage_name!r}") from None


def registered_stages() -> frozenset[str]:
    return frozenset(_HANDLERS)
