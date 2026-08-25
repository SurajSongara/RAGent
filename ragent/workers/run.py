"""Stage consumer.

    python -m ragent.workers.run --pool parse
    python -m ragent.workers.run --queues chunk,embed --concurrency 8

One channel per stage queue, because prefetch is a channel-level setting in AMQP
and the stages want very different values: OCR takes one message at a time to
bound memory, while embed happily holds eight in flight waiting on the network.

Failure handling is deliberately explicit rather than leaning on `reject()`.
Dead-lettering by rejection discards *why* it happened, and a DLQ full of
messages with no error attached is close to useless at 2am — so permanent
failures are published to the DLX with the reason in the headers.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
from typing import Any

from ragent.config import get_settings
from ragent.pipeline.handlers import HandlerNotRegistered, get_handler
from ragent.pipeline.messages import MalformedMessageError, StageMessage
from ragent.pipeline.runner import DeadLetter, Retry, document_ready, next_messages, plan_failure
from ragent.pipeline.stages import PIPELINE, STAGES_BY_NAME, Stage, WorkerPool, validate_pipeline
from ragent.pipeline.topology import DLX_EXCHANGE, INGEST_EXCHANGE, declare

log = logging.getLogger("ragent.worker")


def select_stages(pool: str | None, queues: str | None) -> tuple[Stage, ...]:
    if queues:
        wanted = [q.strip() for q in queues.split(",") if q.strip()]
        unknown = [q for q in wanted if q not in STAGES_BY_NAME]
        if unknown:
            raise SystemExit(
                f"unknown stage(s) {unknown}; expected some of {sorted(STAGES_BY_NAME)}"
            )
        return tuple(STAGES_BY_NAME[q] for q in wanted)
    if pool:
        try:
            selected = WorkerPool(pool)
        except ValueError:
            raise SystemExit(
                f"unknown pool {pool!r}; expected one of {[p.value for p in WorkerPool]}"
            ) from None
        return tuple(s for s in PIPELINE if s.pool is selected)
    return PIPELINE


class StageConsumer:
    def __init__(self, connection: Any, store: Any, stage: Stage) -> None:
        self._connection = connection
        self._store = store
        self._stage = stage

    async def start(self) -> None:
        import aio_pika

        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=self._stage.prefetch)

        self._ingest = await self._channel.get_exchange(INGEST_EXCHANGE)
        self._dlx = await self._channel.get_exchange(DLX_EXCHANGE)
        self._aio_pika = aio_pika

        queue = await self._channel.get_queue(self._stage.queue)
        await queue.consume(self._on_message)
        log.info("consuming %s (prefetch=%d)", self._stage.queue, self._stage.prefetch)

    async def _publish(self, exchange: Any, message: StageMessage, **headers: Any) -> None:
        await exchange.publish(
            self._aio_pika.Message(
                body=message.to_bytes(),
                content_type="application/json",
                delivery_mode=self._aio_pika.DeliveryMode.PERSISTENT,
                headers={"x-trace-id": message.trace_id, **headers},
            ),
            routing_key=STAGES_BY_NAME[message.stage].routing_key,
        )

    async def _on_message(self, raw: Any) -> None:
        # No `requeue=True` anywhere in here. Requeuing puts the message straight
        # back at the head of the same queue, which turns a deterministic failure
        # into a hot loop that starves every other document.
        async with raw.process(requeue=False, ignore_processed=True):
            try:
                message = StageMessage.from_bytes(raw.body)
            except MalformedMessageError as exc:
                log.error("undecodable message on %s: %s", self._stage.queue, exc)
                await raw.reject(requeue=False)
                return

            try:
                handler = get_handler(message.stage)
            except HandlerNotRegistered as exc:
                await self._dead_letter(message, str(exc))
                await raw.ack()
                return

            await self._store.begin_stage(message.run_id, message.stage, message.attempt)

            try:
                metrics = await handler(message)
            except Exception as exc:  # noqa: BLE001 - the whole point is to classify it
                await self._handle_failure(raw, message, exc)
                return

            snapshot = await self._store.complete_stage(
                message.run_id, message.stage, metrics or {}
            )
            followers = next_messages(message, snapshot.completed, snapshot.dispatched)

            # Reserve before publishing. If we crash between the two, the stage
            # is re-queued on resume; if we published first and crashed, a
            # concurrent sibling could publish the same follower again.
            if followers:
                await self._store.mark_dispatched(message.run_id, tuple(f.stage for f in followers))
                for follower in followers:
                    await self._publish(self._ingest, follower)

            if document_ready(message.family, snapshot.completed):
                log.info("document %s ready", message.document_id)

            await raw.ack()

    async def _handle_failure(self, raw: Any, message: StageMessage, exc: Exception) -> None:
        action = plan_failure(message, exc)

        if isinstance(action, Retry):
            await self._store.fail_stage(
                message.run_id, message.stage, f"{type(exc).__name__}: {exc}", permanent=False
            )
            retry_exchange = await self._channel.get_exchange(action.exchange)
            await self._publish(
                retry_exchange,
                message.retried(),
                **{"x-retry-attempt": action.attempt, "x-retry-reason": repr(exc)[:512]},
            )
            log.warning(
                "stage %s attempt %d failed (%s); retrying in %dms",
                message.stage,
                message.attempt,
                exc,
                action.delay_ms,
            )
        else:
            assert isinstance(action, DeadLetter)
            await self._store.fail_stage(
                message.run_id, message.stage, action.reason, permanent=True
            )
            await self._dead_letter(message, action.reason)
            log.error("stage %s dead-lettered: %s", message.stage, action.reason)

        await raw.ack()

    async def _dead_letter(self, message: StageMessage, reason: str) -> None:
        await self._publish(self._dlx, message, **{"x-death-reason": reason[:512]})


async def serve(stages: tuple[Stage, ...]) -> None:
    import aio_pika

    # Importing the package is what populates the handler registry.
    import ragent.ingest.stages  # noqa: F401
    from ragent.pipeline.store_pg import PostgresStageStore

    settings = get_settings()
    validate_pipeline()

    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    setup_channel = await connection.channel()
    await declare(setup_channel)
    await setup_channel.close()

    store = PostgresStageStore()

    consumers = [StageConsumer(connection, store, stage) for stage in stages]
    for consumer in consumers:
        await consumer.start()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):  # not available on Windows
            loop.add_signal_handler(sig, stop.set)

    log.info("worker ready: %s", ", ".join(s.name for s in stages))
    await stop.wait()
    log.info("shutting down")
    await connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragent.workers.run")
    parser.add_argument("--pool", help=f"one of {[p.value for p in WorkerPool]}")
    parser.add_argument("--queues", help="comma-separated stage names, overrides --pool")
    parser.add_argument("--concurrency", type=int, default=1, help="asyncio tasks per stage")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )

    stages = select_stages(args.pool, args.queues)
    asyncio.run(serve(stages))


if __name__ == "__main__":
    main()
