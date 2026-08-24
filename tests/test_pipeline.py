"""The ingest DAG: routing, scheduling, retries and resume.

These are the tests that would otherwise require a broker, a database and a
deliberately crashed worker to exercise. Keeping the decisions pure means the
nasty cases — a fan-in publishing twice, a poison message burning every retry
tier, a document stuck at 90% because a dependency never applied to its format
— are ordinary unit tests.
"""

from __future__ import annotations

import pytest

from ragent.ingest.formats import FormatFamily, UnsupportedFormatError
from ragent.pipeline.handlers import HandlerNotRegistered, get_handler, handler
from ragent.pipeline.messages import MalformedMessageError, StageMessage
from ragent.pipeline.runner import (
    DeadLetter,
    PermanentError,
    Retry,
    document_ready,
    next_messages,
    plan_failure,
    resume_plan,
)
from ragent.pipeline.stages import (
    PIPELINE,
    RETRY_DELAYS_MS,
    STAGES_BY_NAME,
    TERMINAL_STAGE,
    WorkerPool,
    effective_deps,
    is_complete,
    ready_stages,
    retry_delay_ms,
    stage_names_for,
    stages_for,
    validate_pipeline,
)
from ragent.pipeline.store import InMemoryStageStore
from ragent.pipeline.topology import (
    DLX_EXCHANGE,
    INGEST_EXCHANGE,
    build_topology,
    retry_exchange_name,
    tier_label,
)


def msg(stage: str, family: FormatFamily = FormatFamily.PDF, attempt: int = 1) -> StageMessage:
    return StageMessage(
        document_id="doc-1",
        run_id="run-1",
        stage=stage,
        family=family,
        pipeline_version="v1",
        attempt=attempt,
    )


# ---------------------------------------------------------------- graph shape


class TestPipelineShape:
    def test_pipeline_validates(self) -> None:
        validate_pipeline()

    def test_every_family_reaches_the_terminal_stage(self) -> None:
        for family in FormatFamily:
            assert TERMINAL_STAGE in stage_names_for(family)

    def test_pdf_skips_conversion(self) -> None:
        assert "convert" not in stage_names_for(FormatFamily.PDF)

    def test_office_converts_before_parsing(self) -> None:
        path = [s.name for s in stages_for(FormatFamily.OFFICE)]
        assert path.index("convert") < path.index("parse")

    def test_flow_has_no_paged_stages(self) -> None:
        """Markdown has no pages to OCR, no tables to detect, no figures to caption."""
        flow = stage_names_for(FormatFamily.FLOW)
        assert flow.isdisjoint({"parse", "ocr", "tables", "figures", "convert"})
        assert "parse_flow" in flow

    def test_flow_still_chunks_and_embeds(self) -> None:
        flow = stage_names_for(FormatFamily.FLOW)
        assert {"chunk", "contextualize", "embed"} <= flow

    def test_declaration_order_is_topological(self) -> None:
        seen: set[str] = set()
        for stage in PIPELINE:
            assert stage.depends_on <= seen
            seen.add(stage.name)

    def test_stages_are_split_across_both_pools(self) -> None:
        pools = {s.pool for s in PIPELINE}
        assert pools == {WorkerPool.PARSE, WorkerPool.ENRICH}


class TestEffectiveDeps:
    def test_inapplicable_dependency_disappears(self) -> None:
        """`parse` depends on `convert`, which never runs for a PDF."""
        parse = STAGES_BY_NAME["parse"]
        assert "convert" in parse.depends_on
        assert effective_deps(parse, FormatFamily.PDF) == {"detect"}
        assert effective_deps(parse, FormatFamily.OFFICE) == {"detect", "convert"}

    def test_chunk_fans_in_differently_per_family(self) -> None:
        chunk = STAGES_BY_NAME["chunk"]
        assert effective_deps(chunk, FormatFamily.PDF) == {"tables", "figures"}
        assert effective_deps(chunk, FormatFamily.FLOW) == {"parse_flow"}


# ---------------------------------------------------------------- scheduling


class TestScheduling:
    def test_first_stage_is_detect(self) -> None:
        ready = ready_stages(FormatFamily.PDF, completed=frozenset())
        assert [s.name for s in ready] == ["detect"]

    def test_ocr_fans_out_to_tables_and_figures(self) -> None:
        completed = {"detect", "parse", "ocr"}
        assert {s.name for s in ready_stages(FormatFamily.PDF, completed)} == {
            "tables",
            "figures",
        }

    def test_chunk_waits_for_both_branches(self) -> None:
        completed = {"detect", "parse", "ocr", "tables"}
        ready = {s.name for s in ready_stages(FormatFamily.PDF, completed, dispatched=completed)}
        assert "chunk" not in ready
        assert ready == {"figures"}

    def test_chunk_runs_once_both_branches_finish(self) -> None:
        completed = {"detect", "parse", "ocr", "tables", "figures"}
        ready = {s.name for s in ready_stages(FormatFamily.PDF, completed, dispatched=completed)}
        assert ready == {"chunk"}

    def test_dispatched_prevents_double_publishing(self) -> None:
        """The bug this guards: fan-out siblings re-queueing each other."""
        completed = {"detect", "parse", "ocr"}
        first = ready_stages(FormatFamily.PDF, completed)
        assert len(first) == 2

        dispatched = completed | {s.name for s in first}
        # `tables` finishes; `figures` is already in flight and must not re-publish.
        again = ready_stages(FormatFamily.PDF, completed | {"tables"}, dispatched)
        assert again == ()

    def test_full_pdf_run_reaches_completion(self) -> None:
        completed: set[str] = set()
        dispatched: set[str] = set()
        guard = 0
        while not is_complete(FormatFamily.PDF, completed):
            guard += 1
            assert guard < 50, "pipeline did not converge"
            ready = ready_stages(FormatFamily.PDF, completed, dispatched)
            assert ready, f"stalled with {sorted(completed)}"
            for stage in ready:
                dispatched.add(stage.name)
                completed.add(stage.name)
        assert completed == stage_names_for(FormatFamily.PDF)

    @pytest.mark.parametrize("family", list(FormatFamily))
    def test_no_family_deadlocks(self, family: FormatFamily) -> None:
        completed: set[str] = set()
        while ready := ready_stages(family, completed, completed):
            completed.update(s.name for s in ready)
        assert is_complete(family, completed)

    def test_document_ready_only_at_the_end(self) -> None:
        path = stage_names_for(FormatFamily.FLOW)
        assert not document_ready(FormatFamily.FLOW, path - {"embed"})
        assert document_ready(FormatFamily.FLOW, path)


# ---------------------------------------------------------------- resume


class TestResume:
    def test_interrupted_stage_is_requeued(self) -> None:
        """A worker died mid-parse: the stage has a row but never succeeded."""
        plan = resume_plan(
            FormatFamily.PDF,
            completed={"detect"},
            dispatched={"detect", "parse"},
        )
        assert [s.name for s in plan] == ["parse"]

    def test_resume_from_scratch_starts_at_detect(self) -> None:
        plan = resume_plan(FormatFamily.PDF, completed=set(), dispatched=set())
        assert [s.name for s in plan] == ["detect"]

    def test_resume_does_not_redo_completed_work(self) -> None:
        """The point of persisting DAG state: no re-OCRing 200 pages."""
        plan = resume_plan(
            FormatFamily.PDF,
            completed={"detect", "parse", "ocr"},
            dispatched={"detect", "parse", "ocr"},
        )
        assert {s.name for s in plan} == {"tables", "figures"}

    def test_finished_document_has_nothing_to_resume(self) -> None:
        path = stage_names_for(FormatFamily.FLOW)
        assert resume_plan(FormatFamily.FLOW, completed=path, dispatched=path) == ()


# ---------------------------------------------------------------- failures


class TestRetryPolicy:
    def test_backoff_climbs_through_the_tiers(self) -> None:
        stage = STAGES_BY_NAME["embed"]  # max_attempts=5
        delays = [retry_delay_ms(stage, a) for a in range(1, stage.max_attempts)]
        assert delays == [5_000, 30_000, 300_000, 300_000]

    def test_final_attempt_gives_up(self) -> None:
        stage = STAGES_BY_NAME["parse"]
        assert retry_delay_ms(stage, stage.max_attempts) is None

    def test_convert_gives_up_early(self) -> None:
        """LibreOffice failures are rarely transient; do not tie up the pool."""
        assert STAGES_BY_NAME["convert"].max_attempts == 2

    def test_attempt_must_be_one_based(self) -> None:
        with pytest.raises(ValueError, match="1-based"):
            retry_delay_ms(STAGES_BY_NAME["parse"], 0)


class TestFailurePlanning:
    def test_transient_error_retries(self) -> None:
        action = plan_failure(msg("parse"), TimeoutError("broker hiccup"))
        assert isinstance(action, Retry)
        assert action.delay_ms == 5_000
        assert action.attempt == 2
        assert action.exchange == "ragent.retry.5s"

    def test_second_failure_backs_off_further(self) -> None:
        action = plan_failure(msg("parse", attempt=2), TimeoutError("again"))
        assert isinstance(action, Retry)
        assert action.delay_ms == 30_000

    def test_exhausted_attempts_dead_letter(self) -> None:
        stage = STAGES_BY_NAME["parse"]
        action = plan_failure(msg("parse", attempt=stage.max_attempts), TimeoutError("nope"))
        assert isinstance(action, DeadLetter)
        assert "exhausted" in action.reason

    @pytest.mark.parametrize(
        "exc",
        [
            PermanentError("encrypted"),
            UnsupportedFormatError("not a document"),
            MalformedMessageError("bad json"),
        ],
    )
    def test_permanent_errors_skip_retries_entirely(self, exc: Exception) -> None:
        """An encrypted PDF is still encrypted in five minutes."""
        action = plan_failure(msg("parse"), exc)
        assert isinstance(action, DeadLetter)
        assert type(exc).__name__ in action.reason

    def test_unknown_stage_dead_letters(self) -> None:
        action = plan_failure(msg("nonexistent"), RuntimeError("x"))
        assert isinstance(action, DeadLetter)
        assert "unknown stage" in action.reason


# ---------------------------------------------------------------- handoff


class TestNextMessages:
    def test_carries_identity_forward(self) -> None:
        followers = next_messages(msg("detect"), completed={"detect"}, dispatched={"detect"})
        assert [f.stage for f in followers] == ["parse"]
        assert followers[0].document_id == "doc-1"
        assert followers[0].run_id == "run-1"
        assert followers[0].family is FormatFamily.PDF

    def test_resets_attempt_but_keeps_trace(self) -> None:
        """One document is one trace, however many retries happen inside it."""
        source = msg("detect", attempt=3)
        follower = next_messages(source, {"detect"}, {"detect"})[0]
        assert follower.attempt == 1
        assert follower.trace_id == source.trace_id

    def test_flow_routes_to_parse_flow(self) -> None:
        followers = next_messages(
            msg("detect", FormatFamily.FLOW), completed={"detect"}, dispatched={"detect"}
        )
        assert [f.stage for f in followers] == ["parse_flow"]

    def test_office_routes_to_convert(self) -> None:
        followers = next_messages(
            msg("detect", FormatFamily.OFFICE), completed={"detect"}, dispatched={"detect"}
        )
        assert [f.stage for f in followers] == ["convert"]

    def test_terminal_stage_has_no_followers(self) -> None:
        path = stage_names_for(FormatFamily.PDF)
        assert next_messages(msg("embed"), completed=path, dispatched=path) == ()


# ---------------------------------------------------------------- envelope


class TestStageMessage:
    def test_round_trip(self) -> None:
        original = msg("parse")
        assert StageMessage.from_bytes(original.to_bytes()) == original

    def test_serialisation_is_stable(self) -> None:
        """Sorted keys keep one message's body byte-identical across encodings."""
        message = msg("parse")
        assert message.to_bytes() == message.to_bytes()

    def test_each_message_gets_its_own_trace_by_default(self) -> None:
        assert msg("parse").trace_id != msg("parse").trace_id

    def test_rejects_non_json(self) -> None:
        with pytest.raises(MalformedMessageError, match="undecodable"):
            StageMessage.from_bytes(b"\xff\xfe not json")

    def test_rejects_a_json_array(self) -> None:
        with pytest.raises(MalformedMessageError, match="expected an object"):
            StageMessage.from_bytes(b"[1,2,3]")

    def test_rejects_missing_fields(self) -> None:
        with pytest.raises(MalformedMessageError, match="missing fields"):
            StageMessage.from_bytes(b'{"document_id":"d"}')

    def test_rejects_unknown_family(self) -> None:
        body = msg("parse").to_bytes().replace(b'"pdf"', b'"hologram"')
        with pytest.raises(MalformedMessageError, match="unknown family"):
            StageMessage.from_bytes(body)

    def test_rejects_bad_attempt(self) -> None:
        body = msg("parse").to_bytes().replace(b'"attempt":1', b'"attempt":0')
        with pytest.raises(MalformedMessageError, match="positive int"):
            StageMessage.from_bytes(body)

    def test_retried_increments(self) -> None:
        assert msg("parse").retried().attempt == 2

    def test_carries_no_document_content(self) -> None:
        """Messages name documents; they never carry them."""
        body = msg("parse").to_bytes()
        assert len(body) < 512


# ---------------------------------------------------------------- topology


class TestTopology:
    def test_every_stage_has_a_queue_and_a_dlq(self) -> None:
        topology = build_topology()
        names = {q.name for q in topology.queues}
        for stage in PIPELINE:
            assert stage.queue in names
            assert stage.dlq in names

    def test_work_queues_dead_letter_to_the_dlx(self) -> None:
        topology = build_topology()
        for stage in PIPELINE:
            spec = topology.queue(stage.queue)
            assert spec.arguments["x-dead-letter-exchange"] == DLX_EXCHANGE

    def test_dlq_binding_matches_the_stage_routing_key(self) -> None:
        """DLX preserves the routing key, so each stage lands in its own DLQ."""
        topology = build_topology()
        for stage in PIPELINE:
            spec = topology.queue(stage.dlq)
            assert spec.exchange == DLX_EXCHANGE
            assert spec.routing_key == stage.routing_key
            assert spec.consumed is False

    def test_one_retry_exchange_per_tier(self) -> None:
        """A tier needs its own exchange so the message keeps its routing key."""
        topology = build_topology()
        exchanges = {e.name: e for e in topology.exchanges}
        for delay in RETRY_DELAYS_MS:
            name = retry_exchange_name(delay)
            assert exchanges[name].type == "fanout"
            spec = topology.queue(name)
            assert spec.arguments["x-message-ttl"] == delay
            assert spec.arguments["x-dead-letter-exchange"] == INGEST_EXCHANGE
            assert spec.consumed is False

    def test_retry_queues_have_no_dead_letter_routing_key(self) -> None:
        """Setting one would pin every retry to a single stage."""
        topology = build_topology()
        for delay in RETRY_DELAYS_MS:
            spec = topology.queue(retry_exchange_name(delay))
            assert "x-dead-letter-routing-key" not in spec.arguments

    def test_prefetch_reflects_stage_cost(self) -> None:
        topology = build_topology()
        assert topology.queue(STAGES_BY_NAME["ocr"].queue).prefetch == 1
        assert topology.queue(STAGES_BY_NAME["detect"].queue).prefetch > 1

    def test_work_queues_exclude_storage_queues(self) -> None:
        topology = build_topology()
        assert {q.name for q in topology.work_queues} == {s.queue for s in PIPELINE}

    @pytest.mark.parametrize(
        ("ms", "expected"), [(5_000, "5s"), (30_000, "30s"), (300_000, "5m"), (250, "250ms")]
    )
    def test_tier_labels(self, ms: int, expected: str) -> None:
        assert tier_label(ms) == expected


# ---------------------------------------------------------------- handlers


class TestHandlerRegistry:
    def test_unregistered_stage_raises(self) -> None:
        with pytest.raises(HandlerNotRegistered, match="no handler registered"):
            get_handler("parse")

    def test_registration_rejects_a_typo(self) -> None:
        with pytest.raises(ValueError, match="unknown stage"):
            handler("parze")


# ---------------------------------------------------------------- store


class TestInMemoryStore:
    async def test_completion_is_visible_in_the_snapshot(self) -> None:
        store = InMemoryStageStore()
        await store.begin_stage("run-1", "detect", 1)
        snapshot = await store.complete_stage("run-1", "detect")
        assert snapshot.completed == {"detect"}
        assert snapshot.dispatched == {"detect"}

    async def test_dispatched_tracks_queued_but_unfinished_work(self) -> None:
        store = InMemoryStageStore()
        await store.complete_stage("run-1", "detect")
        await store.mark_dispatched("run-1", ("parse",))
        snapshot = await store.snapshot("run-1")
        assert snapshot.in_flight == {"parse"}

    async def test_runs_are_isolated(self) -> None:
        store = InMemoryStageStore()
        await store.complete_stage("run-1", "detect")
        assert (await store.snapshot("run-2")).completed == frozenset()

    async def test_permanent_failure_is_not_rescheduled(self) -> None:
        store = InMemoryStageStore()
        await store.complete_stage("run-1", "detect")
        await store.fail_stage("run-1", "parse", "encrypted", permanent=True)
        snapshot = await store.snapshot("run-1")
        assert ready_stages(FormatFamily.PDF, snapshot.completed, snapshot.dispatched) == ()
        assert store.errors("run-1")["parse"] == "encrypted"
