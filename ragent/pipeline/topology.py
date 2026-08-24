"""RabbitMQ topology, described as data before it is declared against a broker.

Building the topology as a plain value means it can be asserted in unit tests
and printed for review without a broker running. `declare()` is the only part
that needs a live connection.

Retry uses the standard TTL-and-dead-letter trick, with one wrinkle worth
knowing. A delayed message must come back to *its own* stage queue, and
`x-dead-letter-routing-key` is fixed per queue, so it cannot carry a per-message
destination. The fix is one retry exchange per delay tier: the message keeps its
original routing key throughout, and the tier is chosen by which exchange it is
published to.

    ragent.ingest ──► ragent.<stage> ──► consumer
          ▲                 │
          │                 ├─ transient failure, attempts left
          │                 │      └─► ragent.retry.<tier> ──► queue with TTL ──┐
          └─────────────────┴────────────────────────────────────────────────────┘
                            │
                            └─ permanent failure or attempts exhausted
                                   └─► ragent.dlx ──► ragent.<stage>.dlq
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragent.pipeline.stages import PIPELINE, RETRY_DELAYS_MS, Stage

__all__ = [
    "ExchangeSpec",
    "QueueSpec",
    "Topology",
    "build_topology",
    "retry_exchange_name",
    "tier_label",
    "INGEST_EXCHANGE",
    "DLX_EXCHANGE",
]

INGEST_EXCHANGE = "ragent.ingest"
DLX_EXCHANGE = "ragent.dlx"


def tier_label(delay_ms: int) -> str:
    """Human-readable suffix for a delay tier, e.g. 300000 -> '5m'."""
    if delay_ms % 3_600_000 == 0:
        return f"{delay_ms // 3_600_000}h"
    if delay_ms % 60_000 == 0:
        return f"{delay_ms // 60_000}m"
    if delay_ms % 1_000 == 0:
        return f"{delay_ms // 1_000}s"
    return f"{delay_ms}ms"


def retry_exchange_name(delay_ms: int) -> str:
    return f"ragent.retry.{tier_label(delay_ms)}"


@dataclass(frozen=True, slots=True)
class ExchangeSpec:
    name: str
    type: str = "topic"
    durable: bool = True


@dataclass(frozen=True, slots=True)
class QueueSpec:
    name: str
    exchange: str
    routing_key: str
    durable: bool = True
    prefetch: int = 4
    arguments: dict[str, Any] = field(default_factory=dict)
    #: False for retry and dead-letter queues, which are storage, not work.
    consumed: bool = True


@dataclass(frozen=True, slots=True)
class Topology:
    exchanges: tuple[ExchangeSpec, ...]
    queues: tuple[QueueSpec, ...]

    def queue(self, name: str) -> QueueSpec:
        for spec in self.queues:
            if spec.name == name:
                return spec
        raise KeyError(name)

    @property
    def work_queues(self) -> tuple[QueueSpec, ...]:
        return tuple(q for q in self.queues if q.consumed)


def build_topology(
    pipeline: tuple[Stage, ...] = PIPELINE,
    retry_delays: tuple[int, ...] = RETRY_DELAYS_MS,
) -> Topology:
    exchanges: list[ExchangeSpec] = [
        ExchangeSpec(INGEST_EXCHANGE),
        ExchangeSpec(DLX_EXCHANGE),
    ]
    queues: list[QueueSpec] = []

    for stage in pipeline:
        queues.append(
            QueueSpec(
                name=stage.queue,
                exchange=INGEST_EXCHANGE,
                routing_key=stage.routing_key,
                prefetch=stage.prefetch,
                # Rejected messages land on the DLX keeping their routing key,
                # which is what routes them to this stage's own DLQ.
                arguments={"x-dead-letter-exchange": DLX_EXCHANGE},
            )
        )
        queues.append(
            QueueSpec(
                name=stage.dlq,
                exchange=DLX_EXCHANGE,
                routing_key=stage.routing_key,
                consumed=False,
            )
        )

    for delay_ms in retry_delays:
        name = retry_exchange_name(delay_ms)
        exchanges.append(ExchangeSpec(name, type="fanout"))
        queues.append(
            QueueSpec(
                name=name,
                exchange=name,
                # Fanout ignores the key; the message keeps its own for the
                # return trip, which is the entire point of a tier per exchange.
                routing_key="",
                consumed=False,
                arguments={
                    "x-message-ttl": delay_ms,
                    "x-dead-letter-exchange": INGEST_EXCHANGE,
                },
            )
        )

    return Topology(tuple(exchanges), tuple(queues))


async def declare(channel: Any, topology: Topology | None = None) -> None:
    """Declare the topology on a live aio-pika channel. Idempotent."""
    import aio_pika

    topology = topology or build_topology()
    declared: dict[str, Any] = {}

    for spec in topology.exchanges:
        declared[spec.name] = await channel.declare_exchange(
            spec.name,
            aio_pika.ExchangeType(spec.type),
            durable=spec.durable,
        )

    for spec in topology.queues:
        queue = await channel.declare_queue(
            spec.name,
            durable=spec.durable,
            arguments=spec.arguments or None,
        )
        await queue.bind(declared[spec.exchange], routing_key=spec.routing_key)
