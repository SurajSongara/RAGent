# RAGent

[![CI](https://github.com/SurajSongara/RAGent/actions/workflows/ci.yml/badge.svg)](https://github.com/SurajSongara/RAGent/actions/workflows/ci.yml)

A document intelligence prototype for asynchronous ingestion, hybrid retrieval,
and answers with citations linked to source regions or character ranges.
Built with Python, FastAPI, RabbitMQ, PostgreSQL, and Qdrant.

> **Status: in development.** The ingestion and retrieval foundation is implemented.
> OCR recognition, table structure, and figure captioning remain stubbed.
> MCP integration, the agent loop, and the evaluation harness are planned.
> There is no hosted demo or published retrieval-quality benchmark yet.

## What works and what is planned

| Area | Current status |
|---|---|
| Ingestion | Format routing, persistent DAG state, retries, and dead-letter handling implemented. OCR/table/figure enrichment remains unfinished. |
| Retrieval | Dense and lexical retrieval with reciprocal-rank fusion; four chunking strategies implemented. Comparative quality results are not yet published. |
| Answers and UI | Upload, pipeline progress, streaming responses, and source-linked citations implemented. With no model credentials, answers fall back to extracted passages. |
| Model interface | Anthropic and OpenAI-compatible providers; OpenAI-compatible API endpoints. Individual provider deployments still need their own validation. |
| Evaluation | Golden Q/A corpus, retrieval/generation metrics, benchmark charts, and quality regression gates planned. Current CI checks lint, formatting, and automated tests. |
| Agents and MCP | Bidirectional MCP and a LangGraph reasoning loop planned. |
| Operations | End-to-end OpenTelemetry coverage, cost/latency dashboard, and hosted demonstration planned. |

## Why this exists

Document retrieval needs more than embedding text. This project explores
recoverable ingestion and citations that retain their provenance through the
pipeline. Paged documents use page and bounding-box references; flow text uses
character ranges instead of invented page coordinates.

SEC filings are the intended domain for the evaluation work because they contain
checkable numerical questions and difficult tables. The next step is a versioned
golden dataset and measured comparisons of the four chunking strategies. Those
results are not available yet, so no strategy is presented as the winner.

## Quick start

Two ways to run it, and the split is deliberate.

**Everything in Docker** — one command from a cold clone:

```bash
cp .env.example .env      # works with no API keys: EMBEDDING_BACKEND=local
make up
make seed
```

**Infra in Docker, app native** — the day-to-day loop, with no image in the
edit-run cycle:

```bash
make install              # one-time: venv + dependencies
make dev                  # postgres, qdrant, valkey, rabbitmq, minio
make api                  # terminal 2, reloads on save
make worker               # terminal 3, reloads on save
make web                  # terminal 4
make seed
```

Both modes mount the source and reload on save, so a code change never needs a
rebuild — only a dependency change does.

One stage stays in Docker either way: `convert` needs LibreOffice, which is
~500MB and has no business on your laptop. `make worker` skips that queue, and
`make worker-convert` runs it in a container. Without the skip the native worker
would take convert jobs it cannot service and fail them permanently, so the
container that *could* handle them would never see them.

| | |
|---|---|
| Web UI | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Pipeline queues | http://localhost:15672 |
| Vector store | http://localhost:6333/dashboard |
| Object storage | http://localhost:9001 |

`make help` lists everything else.

## OpenAI compatible, both ways

**Driven by any OpenAI-compatible model.** Anthropic and OpenAI are both
first-class, and "OpenAI-compatible" does real work here — the same code path
reaches OpenAI, Azure OpenAI, xAI (Grok), DeepSeek, Ollama, vLLM, Groq,
Together, OpenRouter and LM Studio, because they differ only in
`OPENAI_BASE_URL`.

```bash
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama; needs no key
OPENAI_MODEL_SYNTHESIS=llama3.2:3b
EMBEDDING_BACKEND=openai
OPENAI_EMBEDDING_MODEL=nomic-embed-text
```

Embedding dimensions are discovered from the vectors, never assumed — pointing
at a model this code has never heard of works, and a genuine mismatch against an
existing index fails with a message that says so.

**Consumed as one.** Point any OpenAI client at `http://localhost:8000/v1` and
RAGent answers as a model, with citations attached:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

r = client.chat.completions.create(
    model="ragent-layout",
    messages=[{"role": "user", "content": "why fuse retrieval on rank?"}],
)
print(r.choices[0].message.content)
print(r.model_extra["citations"])   # page + bbox, or character range
```

That works from Open WebUI, LibreChat, Cursor, LangChain's `ChatOpenAI`, or curl.

The mapping worth noticing: **the model name selects the chunking strategy.**

| Model | Retrieval |
|---|---|
| `ragent` | the configured default |
| `ragent-layout` | layout-aware chunks |
| `ragent-recursive` | separator-hierarchy chunks |
| `ragent-fixed` | structure-blind token windows |
| `ragent-semantic` | embedding-distance chunks |

So the Phase 2 bake-off is drivable from any OpenAI client — change the model in
Open WebUI's dropdown and you are A/B testing retrieval strategies against the
same corpus, with no bespoke UI.


## Format routing and limitations

Detection leads with magic bytes, never the extension — users rename files, and
scanners emit `.tif` files that are really JPEGs. The detected **family** picks
the route through the ingest DAG.

| Family | Formats | Route |
|---|---|---|
| **PDF** | `.pdf` | Native text parsing implemented. Low-confidence pages route to the unfinished OCR stage. |
| **Image** | `.png` `.jpg` `.tiff` `.gif` `.bmp` `.webp` | Detection/routing implemented; usable text extraction is blocked on OCR recognition. |
| **Office** | `.docx` `.xlsx` `.pptx` `.doc` `.xls` `.ppt` `.odt` `.ods` `.odp` `.rtf` | Converted to PDF, then the PDF route |
| **Web** | `.html` `.htm` | Rendered to PDF, then the PDF route |
| **Flow** | `.md` `.txt` `.csv` `.tsv` `.json` `.xml` | No pages or geometry; character-offset provenance |


## Roadmap

**Phase 1 — Foundation**
- [x] Data model with end-to-end bbox provenance
- [x] Stack topology, healthchecked bring-up, native dev loop
- [x] Ingest primitives: selective-OCR gate, PDF coordinate conversion, financial cell parsing, four chunking strategies
- [x] Format detection and routing: PDF, images, Office, HTML, flow text
- [x] Ingest DAG: conditional per-format graph, scheduler, retry/DLQ policy, resume-on-crash
- [x] RabbitMQ topology, stage consumer, Postgres-backed DAG state
- [x] Hybrid retrieval: Qdrant dense + Postgres lexical, RRF fusion
- [x] Provider abstraction: Anthropic and any OpenAI-compatible endpoint
- [x] OpenAI-compatible server: `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`
- [x] Chat API with inline citations, streaming over SSE
- [x] Web UI: upload, live pipeline view, citation viewer
- [ ] Stage handlers still stubbed: OCR recognition, table structure, figure captioning
- [ ] Cross-encoder reranking

**Phase 2 — The proof**
- [ ] Golden Q/A set over the demo corpus
- [ ] Retrieval metrics: recall@k, nDCG, MRR
- [ ] Generation metrics: faithfulness, citation precision
- [ ] Chunking strategy bake-off + published chart
- [ ] CI regression gate

**Phase 3 — The agent**
- [ ] MCP server exposing the corpus as tools
- [ ] MCP client for dynamic external tool use
- [ ] LangGraph plan → retrieve → grade → rewrite → synthesise → verify loop
- [ ] Task-tier model router with fallback

**Phase 4 — Polish**
- [ ] OTel traces across every stage
- [ ] Cost and latency dashboard
- [ ] Live hosted demo

## Architecture

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the four planes, the retrieval
design, the planned MCP integration, and the reasoning behind each choice. The
status table above and roadmap distinguish implementation from design intent.

## License

MIT
