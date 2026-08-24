"""The ingest DAG: what runs, in what order, for which kind of document.

Formats do not share one linear pipeline. A scanned PNG has no text layer to
assess, a `.docx` has to become a PDF before anything can parse it, and a
Markdown file has neither pages nor geometry. Modelling that as a conditional
DAG keeps the branching in one readable place instead of scattering `if
family ==` checks through eight consumers.

    detect ─┬─ OFFICE / WEB ─► convert ─┐
            │                           ├─► parse ─► ocr ─┬─► tables ──┐
            ├─ PDF / IMAGE ─────────────┘                 └─► figures ─┤
            │                                                          │
            └─ FLOW ─► parse_flow ────────────────────────────────────►┤
                                                                       │
                            chunk ─► contextualize ─► embed ◄───────────┘

The trick that keeps this simple: a stage declares every stage it *could*
depend on, and the router intersects those with the stages that actually apply
to this document's family. So `parse` declares a dependency on `convert`, and
for a PDF — where `convert` never runs — that dependency simply disappears
rather than deadlocking.

Because stage completion is persisted per document in `ingest_stages`, resuming
a crashed run is just asking this module what is ready given what finished. No
separate recovery path, and no re-OCRing 200 pages because a worker died on
page 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ragent.ingest.formats import FormatFamily

__all__ = [
    "Stage",
    "WorkerPool",
    "PIPELINE",
    "STAGES_BY_NAME",
    "RETRY_DELAYS_MS",
    "stages_for",
    "stage_names_for",
    "ready_stages",
    "is_complete",
    "retry_delay_ms",
    "validate_pipeline",
]

PAGED = frozenset({FormatFamily.PDF, FormatFamily.IMAGE, FormatFamily.OFFICE, FormatFamily.WEB})
ALL_FAMILIES = frozenset(FormatFamily)

#: Shared backoff tiers. One retry queue per tier rather than per stage, since
#: the delay is the only thing that differs and queues are not free.
RETRY_DELAYS_MS: tuple[int, ...] = (5_000, 30_000, 300_000)


class WorkerPool(StrEnum):
    """Which consumer process runs a stage.

    Split by workload shape, not by pipeline position: parsing and OCR are
    CPU-bound and slow, while the LLM stages are latency-bound and rate-limited.
    Pooling them together would let a 200-page OCR job starve the embed queue.
    """

    PARSE = "parse"
    ENRICH = "enrich"


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    pool: WorkerPool
    families: frozenset[FormatFamily]
    depends_on: frozenset[str]
    #: Total attempts before the message is dead-lettered for inspection.
    max_attempts: int = 3
    #: Consumer prefetch. Low for memory-hungry stages, higher for IO-bound ones.
    prefetch: int = 4
    description: str = ""

    @property
    def queue(self) -> str:
        return f"ragent.{self.name}"

    @property
    def routing_key(self) -> str:
        return f"ingest.{self.name}"

    @property
    def dlq(self) -> str:
        return f"ragent.{self.name}.dlq"

    def applies_to(self, family: FormatFamily) -> bool:
        return family in self.families


PIPELINE: tuple[Stage, ...] = (
    Stage(
        name="detect",
        pool=WorkerPool.PARSE,
        families=ALL_FAMILIES,
        depends_on=frozenset(),
        prefetch=16,
        description="Hash for dedupe, sniff the format, record the document.",
    ),
    Stage(
        name="convert",
        pool=WorkerPool.PARSE,
        families=frozenset({FormatFamily.OFFICE, FormatFamily.WEB}),
        depends_on=frozenset({"detect"}),
        # LibreOffice is the flakiest thing in the stack and failures are rarely
        # transient, so we give up quickly rather than tie up the pool.
        max_attempts=2,
        prefetch=1,
        description="Render Office/HTML to PDF so the paged path can take over.",
    ),
    Stage(
        name="parse",
        pool=WorkerPool.PARSE,
        families=PAGED,
        depends_on=frozenset({"detect", "convert"}),
        prefetch=2,
        description="Layout extraction: blocks, reading order, bboxes, sections.",
    ),
    Stage(
        name="parse_flow",
        pool=WorkerPool.PARSE,
        families=frozenset({FormatFamily.FLOW}),
        depends_on=frozenset({"detect"}),
        prefetch=8,
        description="Text extraction with character offsets; no pages, no geometry.",
    ),
    Stage(
        name="ocr",
        pool=WorkerPool.PARSE,
        families=PAGED,
        depends_on=frozenset({"parse"}),
        prefetch=1,
        description="Selective OCR: only regions the text-layer gate flagged.",
    ),
    Stage(
        name="tables",
        pool=WorkerPool.PARSE,
        families=PAGED,
        depends_on=frozenset({"parse", "ocr"}),
        prefetch=2,
        description="Table structure to typed cells, not flattened markdown.",
    ),
    Stage(
        name="figures",
        pool=WorkerPool.ENRICH,
        families=PAGED,
        depends_on=frozenset({"parse", "ocr"}),
        max_attempts=4,
        prefetch=4,
        description="Vision-model captions for charts and diagrams.",
    ),
    Stage(
        name="chunk",
        pool=WorkerPool.ENRICH,
        families=ALL_FAMILIES,
        depends_on=frozenset({"tables", "figures", "parse_flow"}),
        prefetch=8,
        description="Apply every configured chunking strategy.",
    ),
    Stage(
        name="contextualize",
        pool=WorkerPool.ENRICH,
        families=ALL_FAMILIES,
        depends_on=frozenset({"chunk"}),
        # Provider rate limits are the usual failure here and they do clear.
        max_attempts=5,
        prefetch=8,
        description="Write the one-line locating preamble for each chunk.",
    ),
    Stage(
        name="embed",
        pool=WorkerPool.ENRICH,
        families=ALL_FAMILIES,
        depends_on=frozenset({"contextualize"}),
        max_attempts=5,
        prefetch=4,
        description="Embed and upsert into Qdrant; index lexically in Postgres.",
    ),
)

STAGES_BY_NAME: dict[str, Stage] = {s.name: s for s in PIPELINE}

#: Reaching this stage means the document is ready to answer questions.
TERMINAL_STAGE = "embed"


def stages_for(family: FormatFamily) -> tuple[Stage, ...]:
    """Stages this family runs, in declaration (topological) order."""
    return tuple(s for s in PIPELINE if s.applies_to(family))


def stage_names_for(family: FormatFamily) -> frozenset[str]:
    return frozenset(s.name for s in stages_for(family))


def effective_deps(stage: Stage, family: FormatFamily) -> frozenset[str]:
    """Dependencies that actually exist for this family.

    A stage declares every predecessor it could have across all formats. Only
    the ones on this family's path are real — which is what lets `parse` depend
    on `convert` without stalling PDFs forever.
    """
    return stage.depends_on & stage_names_for(family)


def ready_stages(
    family: FormatFamily,
    completed: frozenset[str] | set[str],
    dispatched: frozenset[str] | set[str] = frozenset(),
) -> tuple[Stage, ...]:
    """Stages whose dependencies are satisfied and which are not already in flight.

    This is the whole scheduler. Restarting a half-finished document means
    loading its stage rows from `ingest_stages` and calling this.

    `dispatched` is what prevents double-publishing where the graph fans out and
    back in. `tables` and `figures` both become ready when `ocr` finishes, and
    both feed `chunk`; without tracking what has already been queued, whichever
    of them finished second would publish `figures` — or `chunk` — a second time.
    Pass every stage that has a row, in any status; pass only succeeded ones as
    `completed`.
    """
    completed = frozenset(completed)
    blocked = completed | frozenset(dispatched)
    out = []
    for stage in stages_for(family):
        if stage.name in blocked:
            continue
        if effective_deps(stage, family) <= completed:
            out.append(stage)
    return tuple(out)


def is_complete(family: FormatFamily, completed: frozenset[str] | set[str]) -> bool:
    return stage_names_for(family) <= frozenset(completed)


def retry_delay_ms(stage: Stage, attempt: int) -> int | None:
    """Backoff for the next attempt, or None when the message should be dead-lettered.

    `attempt` is 1-based and counts attempts already made.
    """
    if attempt < 1:
        raise ValueError(f"attempt is 1-based, got {attempt}")
    if attempt >= stage.max_attempts:
        return None
    return RETRY_DELAYS_MS[min(attempt - 1, len(RETRY_DELAYS_MS) - 1)]


def validate_pipeline() -> None:
    """Structural checks. Called by the tests and at worker startup.

    Catches the mistakes that would otherwise show up as documents silently
    stuck at 90%: a typo'd dependency, a cycle, an unreachable stage, or a
    family whose path never reaches the terminal stage.
    """
    names = {s.name for s in PIPELINE}

    for stage in PIPELINE:
        unknown = stage.depends_on - names
        if unknown:
            raise ValueError(f"stage {stage.name!r} depends on unknown {sorted(unknown)}")
        if not stage.families:
            raise ValueError(f"stage {stage.name!r} applies to no format family")
        if stage.max_attempts < 1:
            raise ValueError(f"stage {stage.name!r} has max_attempts < 1")

    # Declaration order must already be topological, so a worker can iterate
    # PIPELINE directly without sorting.
    seen: set[str] = set()
    for stage in PIPELINE:
        missing = stage.depends_on - seen
        if missing:
            raise ValueError(
                f"stage {stage.name!r} is declared before its dependencies {sorted(missing)}"
            )
        seen.add(stage.name)

    for family in FormatFamily:
        path = stages_for(family)
        if not path:
            raise ValueError(f"family {family} has no stages")
        if TERMINAL_STAGE not in {s.name for s in path}:
            raise ValueError(f"family {family} never reaches {TERMINAL_STAGE!r}")

        # Simulate a full run: every stage must eventually become ready.
        completed: set[str] = set()
        while True:
            ready = ready_stages(family, completed)
            if not ready:
                break
            completed.update(s.name for s in ready)
        if not is_complete(family, completed):
            stuck = stage_names_for(family) - completed
            raise ValueError(f"family {family} deadlocks; unreachable stages {sorted(stuck)}")


validate_pipeline()
