# RAGent

Document intelligence over SEC filings and whatever else you throw at it —
PDFs, scans and images, Office documents, HTML, plain text. Async ingest
pipeline, hybrid retrieval, answers whose every citation resolves back to the
exact spot in the source it came from, and an MCP interface in both directions.

> **Status: in development, but it runs.** Upload a document and it goes
> through the full pipeline to a cited, clickable answer. Three stages are
> still stubs — OCR recognition, table structure and figure captioning — and
> each is marked NOT-IMPLEMENTED in the source with what it takes to finish.
> See [Roadmap](#roadmap); nothing is claimed to work until its box is ticked.

---

## Why this exists

Most RAG demos are a loop over `text-embedding` + top-k + a prompt. They fall over
on the first real document and they have no way to tell you whether a change made
them better or worse.

This one is built around three things that scaffolding does not produce:

**1. An eval harness with numbers.** A golden Q/A set over real filings,
retrieval scored on recall@k / nDCG / MRR, generation scored on faithfulness and
citation precision, run in CI as a regression gate. Every claim in the docs is
backed by a number the harness produced.

**2. Citations you can see.** Ask a question, get an answer, click a citation, and
the original page renders with the exact bounding box highlighted. That requires
carrying `page + bbox` provenance intact through OCR, chunking, embedding,
retrieval, and generation without dropping it anywhere. Formats with no pages to
highlight — Markdown, CSV, plain text — carry character offsets instead, because
two honest provenance modes beat one that fabricates coordinates.

**3. Benchmarked decisions, not asserted ones.** Four chunking strategies are
implemented and indexed side by side over the same corpus. `make bench` runs them
head to head and produces the chart. The winner is whichever one the data picks.

The corpus is SEC filings (10-K, 10-Q, investor decks) because the answers are
numeric and checkable, the tables are genuinely hard, and EDGAR is free to
redistribute — so the eval is objective rather than vibes.

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


## Supported formats

Detection leads with magic bytes, never the extension — users rename files, and
scanners emit `.tif` files that are really JPEGs. The detected **family** picks
the route through the ingest DAG.

| Family | Formats | Route |
|---|---|---|
| **PDF** | `.pdf` | Parsed natively; only low-confidence pages get OCRed |
| **Image** | `.png` `.jpg` `.tiff` `.gif` `.bmp` `.webp` | No text layer exists, so always OCR |
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
design, the MCP-in-both-directions setup, and the reasoning behind each choice.

## License

MIT
