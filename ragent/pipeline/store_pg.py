"""Postgres-backed stage store.

The one operation that has to be careful is `complete_stage`. Recording a
completion and reading back the run's progress must happen as a single atomic
step, because two stages finishing at the same instant is the normal case here —
`tables` and `figures` run concurrently and both feed `chunk`. A read-then-write
race would either publish `chunk` twice or never publish it at all.

`SELECT ... FOR UPDATE` on the run row serialises completions per document,
which is cheap (contention is only ever between a handful of stages) and removes
the race entirely.
"""

from __future__ import annotations

import uuid
from typing import Any

from ragent.db.pool import acquire
from ragent.pipeline.store import StageSnapshot

__all__ = ["PostgresStageStore"]


class PostgresStageStore:
    async def mark_dispatched(self, run_id: str, stages: tuple[str, ...]) -> None:
        if not stages:
            return
        async with acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO ingest_stages (run_id, stage, status)
                VALUES ($1, $2, 'pending')
                ON CONFLICT (run_id, stage) DO NOTHING
                """,
                [(uuid.UUID(run_id), stage) for stage in stages],
            )

    async def begin_stage(self, run_id: str, stage: str, attempt: int) -> None:
        async with acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingest_stages (run_id, stage, status, attempt, started_at)
                VALUES ($1, $2, 'running', $3, now())
                ON CONFLICT (run_id, stage) DO UPDATE
                    SET status = 'running', attempt = $3, started_at = now()
                """,
                uuid.UUID(run_id),
                stage,
                attempt,
            )

    async def complete_stage(
        self, run_id: str, stage: str, metrics: dict[str, Any] | None = None
    ) -> StageSnapshot:
        async with acquire() as conn, conn.transaction():
            # Serialise concurrent completions for this document.
            await conn.execute(
                "SELECT id FROM ingest_runs WHERE id = $1 FOR UPDATE", uuid.UUID(run_id)
            )
            await conn.execute(
                """
                INSERT INTO ingest_stages (run_id, stage, status, finished_at, metrics)
                VALUES ($1, $2, 'succeeded', now(), $3::jsonb)
                ON CONFLICT (run_id, stage) DO UPDATE
                    SET status = 'succeeded', finished_at = now(),
                        metrics = EXCLUDED.metrics, error = NULL
                """,
                uuid.UUID(run_id),
                stage,
                metrics or {},
            )
            return await self._snapshot(conn, run_id)

    async def fail_stage(self, run_id: str, stage: str, error: str, *, permanent: bool) -> None:
        async with acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingest_stages (run_id, stage, status, finished_at, error)
                VALUES ($1, $2, $3::stage_status, now(), $4)
                ON CONFLICT (run_id, stage) DO UPDATE
                    SET status = EXCLUDED.status,
                        finished_at = now(),
                        error = EXCLUDED.error
                """,
                uuid.UUID(run_id),
                stage,
                # A retryable failure goes back to pending so the resume path
                # picks it up if the retry message is somehow lost; a permanent
                # one stays failed and waits for a human.
                "failed" if permanent else "pending",
                error[:4000],
            )
            if permanent:
                await conn.execute(
                    """
                    UPDATE documents SET status = 'failed'
                     WHERE id = (SELECT document_id FROM ingest_runs WHERE id = $1)
                    """,
                    uuid.UUID(run_id),
                )

    async def snapshot(self, run_id: str) -> StageSnapshot:
        async with acquire() as conn:
            return await self._snapshot(conn, run_id)

    @staticmethod
    async def _snapshot(conn: Any, run_id: str) -> StageSnapshot:
        rows = await conn.fetch(
            "SELECT stage, status FROM ingest_stages WHERE run_id = $1", uuid.UUID(run_id)
        )
        return StageSnapshot(
            completed=frozenset(r["stage"] for r in rows if r["status"] == "succeeded"),
            # Anything with a row has been queued at least once. A stage that
            # failed permanently stays here so the scheduler will not retry it.
            dispatched=frozenset(
                r["stage"]
                for r in rows
                if r["status"] in ("pending", "running", "succeeded", "failed")
            ),
        )
