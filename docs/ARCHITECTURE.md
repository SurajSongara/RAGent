# Architecture

Four planes. Each is independently reviewable, and each answers a different
question a reviewer is actually asking.

| Plane | Question it answers |
|---|---|
| Ingest | Can you handle documents that are genuinely hard, not just clean text PDFs? |
| Retrieval | Do you understand why hybrid + rerank beats cosine similarity on top-k? |
| Agent | Can you build something that plans and self-corrects, not a lookup wrapper? |
| Control | Do you know what your system costs, how fast it is, and whether it regressed? |

---

## 1. Ingest plane

```
upload -> MinIO (raw) -> RabbitMQ topic exchange `ragent.ingest`

    detect -+- OFFICE / WEB -> convert -+
            |                           +-> parse -> ocr -+-> tables --+
            +- PDF / IMAGE -------------+                 +-> figures -+
            |                                                          |
            +- FLOW -> parse_flow ------------------------------------>+
                                                                       |
                            chunk -> contextualize -> embed <----------+

    DAG state in Postgres (ingest_runs / ingest_stages)
    progress -> Valkey pub/sub -> SSE -> live UI
```

Formats do not share one linear pipeline, so the graph is conditional. The
mechanism that keeps that from becoming a mess of `if family ==` checks: a stage
declares every predecessor it *could* have, and the scheduler intersects those
with the stages that actually apply to this document's family. `parse` depends
on `convert`; for a PDF, where `convert` never runs, that dependency simply
disappears instead of deadlocking.

Each stage is an idempotent consumer with its own queue, prefetch, retry policy
and dead-letter queue. Per-document DAG state lives in `ingest_stages`, so
resuming a crashed run is just asking the scheduler what is ready given what
finished — no separate recovery path, and no re-OCRing 200 pages because a
worker died on page 3.

Four decisions here are the ones worth defending in an interview:

**Selective OCR.** Naive pipelines rasterise and OCR every page. A born-digital
10-K already has a perfect text layer; re-OCRing it *destroys* accuracy while
burning minutes per document. We score the embedded text layer per page and only
OCR regions below `OCR_CONFIDENCE_THRESHOLD`. Scanned exhibits get the full
treatment; the other 95% of pages do not. Images skip the gate entirely — they
have no text layer to score.

**Tables as cells, not markdown.** The common shortcut is to flatten a table into
a pipe-delimited blob inside a chunk and hope the model reads it back correctly.
Financial tables break that immediately — merged headers, footnote markers,
parenthesised negatives, "in thousands except per share" units. We extract to
`doc_tables` / `table_cells` with `numeric_value` parsed once at ingest, so
numeric questions resolve against typed data.

**Two provenance modes.** Paged sources carry `page_no` plus a normalised bbox,
and the viewer draws a highlight. Flow sources — Markdown, CSV, plain text — have
no pages and no geometry, so they carry character offsets and the viewer
highlights a text range. The alternative was to lay flow text out onto synthetic
pages so that everything had a bbox, which would mean fabricating coordinates
that correspond to nothing. The `block_is_locatable` constraint enforces that
every block does one or the other.

**Retry topology.** Transient failures go to a delay tier and come back;
permanent ones (encrypted PDF, unsupported content, malformed message) go
straight to the DLQ, because an encrypted PDF is still encrypted in five minutes
and burning three attempts to rediscover that only delays the operator seeing it.
Delayed messages must return to their *own* stage queue, and
`x-dead-letter-routing-key` is fixed per queue, so there is one retry exchange
per delay tier and the message keeps its original routing key throughout.

## 2. Retrieval plane

```
query ──► semantic cache (Valkey)  ── hit ──► done
      └─► ┌ dense: Qdrant, payload-filtered (cik, fiscal_year, form_type)
          └ sparse: Postgres tsvector
                    └─► RRF fusion ──► cross-encoder rerank (50 → 8) ──► context
```

Qdrant over pgvector specifically for **filtered** vector search: "compare FY24
against FY25 for this CIK" is a pre-filtered ANN query, not a fetch-100-and-hope
rerank. Postgres holds the lexical side, which matters more than people expect on
filings — exact ticker symbols, section identifiers like "Item 7A", and defined
terms are precisely where dense retrieval is weakest.

Lexical ranking currently uses `ts_rank_cd`, which is not true BM25. ParadeDB's
`pg_search` is a drop-in upgrade if the eval shows lexical recall is the binding
constraint — deliberately deferred until there is a number saying it matters.

## 3. Agent plane (LangGraph)

```
plan ──► decompose into sub-questions
     ──► retrieve (parallel per sub-question)
     ──► grade relevance
     ──► [insufficient] ─► rewrite query ─┐  (bounded, max 2 loops)
     ◄────────────────────────────────────┘
     ──► synthesise with mandatory inline citations
     ──► verify: is every claim traceable to a chunk? ─► [no] revise
```

Graph state checkpoints to Postgres, so conversations are durable and resumable
and every agent run can be replayed for debugging.

**MCP runs in both directions**, which is the piece almost nobody builds:

- **As a server** — the corpus is exposed as tools (`search_filings`,
  `compare_metrics`, `get_table`, `cite`). Claude Desktop or Claude Code can then
  query this system directly. `make mcp` prints the config to paste in.
- **As a client** — the agent connects out to external MCP servers and selects
  tools dynamically at plan time.

## 4. Control plane

OpenTelemetry spans across every stage and every retrieval hop, exported to
Jaeger. Per-message token and cost accounting lands in `messages`, so the model
router's decisions are auditable after the fact instead of asserted in a README.

Routing is task-tier based: a cheap model grades relevance and classifies, a
strong model synthesises, a vision model captions figures. Provider errors fall
back down the tier.

---

## Deliberate exclusions

**MySQL.** Postgres already serves as system of record, lexical index, table
store, and LangGraph checkpointer. A second RDBMS adds an ops surface and zero
capability. Two databases doing one job reads as resume-padding.

**LangChain (the framework).** LangGraph is used for the agent loop, where cyclic
stateful orchestration genuinely earns its place. The retrieval and ingest code is
hand-written. Chain abstractions would hide the exact mechanics this project
exists to demonstrate, and debugging retrieval through three layers of wrapper is
worse than the 200 lines they replace.

**pgvector.** See the filtered-ANN argument above. It would be the right call for
a smaller corpus with no metadata filtering.

**Kubernetes.** Local `docker compose up` first. Nothing in the design blocks a
Helm chart later; the workers are already split along the axes that would scale
independently.

## Data model

The load-bearing invariant:

```
documents ──► pages ──► blocks ──► chunk_blocks ──► chunks ──► citations
                          │
              paged: page_no + normalised bbox   ──► highlight a region
              flow:  char_start / char_end       ──► highlight a text range
```

Every generated sentence resolves back to its exact location in the source. Any
change that breaks a link in that chain is a bug, regardless of what it improves.

Which of the two locators a block carries depends on its format family, and the
`block_is_locatable` CHECK constraint enforces that it carries one of them. A
block that can name neither a region nor a range is unciteable, and the database
refuses to store it rather than letting it surface later as a citation pointing
nowhere.

bboxes are normalised 0..1 against page dimensions so the viewer highlights
correctly at any zoom without knowing the scale the page was rendered at.

See [`infra/postgres/init.sql`](../infra/postgres/init.sql).
