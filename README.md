# RAGent

Document intelligence over SEC filings. Async ingest pipeline, hybrid retrieval,
answers whose every citation highlights the exact region of the source page it
came from, and an MCP interface in both directions.

> **Status: in development.** The data model and stack topology are in place.
> The pipeline stages are being built now — see [Roadmap](#roadmap) for what
> actually runs today. Nothing below is claimed to work until its box is ticked.

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
the original PDF page renders with the exact bounding box highlighted. That
requires carrying `page + bbox` provenance intact through OCR, chunking,
embedding, retrieval, and generation without dropping it anywhere.

**3. Benchmarked decisions, not asserted ones.** Four chunking strategies are
implemented and indexed side by side over the same corpus. `make bench` runs them
head to head and produces the chart. The winner is whichever one the data picks.

The corpus is SEC filings (10-K, 10-Q, investor decks) because the answers are
numeric and checkable, the tables are genuinely hard, and EDGAR is free to
redistribute — so the eval is objective rather than vibes.

## Quick start

```bash
cp .env.example .env      # works with no API keys: EMBEDDING_BACKEND=local
make up                   # brings up the full stack, waits until healthy
make seed                 # pulls the demo EDGAR corpus and ingests it
```

| | |
|---|---|
| Web UI | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Pipeline queues | http://localhost:15672 |
| Vector store | http://localhost:6333/dashboard |
| Traces | http://localhost:16686 |

`make help` lists everything else.

## Stack

| | | |
|---|---|---|
| **Postgres** | System of record, lexical index, extracted tables, agent checkpoints | Earns its place four times over |
| **Qdrant** | Dense vectors with payload filters | Filtered ANN, not fetch-and-rerank |
| **Valkey** | Semantic cache, SSE pub/sub, rate limits, locks | |
| **RabbitMQ** | Per-stage queues with DLQs | Right tool for a multi-stage DAG |
| **MinIO** | Raw PDFs and page renders | S3-compatible |
| **LangGraph** | Agent loop, checkpointed to Postgres | Cyclic and stateful, so it earns the dependency |
| **FastAPI + SSE** | Streaming tokens and live pipeline progress | |
| **Next.js** | Chat, PDF.js citation viewer, pipeline view, eval dashboard | |

What was deliberately left out, and why, is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#deliberate-exclusions).

## Roadmap

**Phase 1 — Foundation**
- [x] Data model with end-to-end bbox provenance
- [x] Stack topology, healthchecked one-command bring-up
- [ ] Async ingest DAG: classify → parse → selective OCR → tables → figures → chunk → contextualize → embed
- [ ] Hybrid retrieval: Qdrant + Postgres lexical, RRF fusion, cross-encoder rerank
- [ ] Chat API with inline citations
- [ ] Web UI with PDF.js bbox highlighting

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
