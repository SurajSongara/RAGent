"""An OpenAI-compatible surface, so RAGent can be used *as* a model.

Point any OpenAI client at `http://localhost:8000/v1` and it works: the `openai`
SDK, Open WebUI, LibreChat, Cursor, LangChain's ChatOpenAI, curl. What comes back
is a RAG-grounded, cited answer rather than a raw model completion.

The mapping that makes this worth building: **the model name selects the
chunking strategy.**

    ragent              the configured default
    ragent-layout       layout-aware chunks
    ragent-recursive    separator-hierarchy chunks
    ragent-fixed        structure-blind token windows
    ragent-semantic     embedding-distance chunks

So the Phase 2 bake-off is drivable from any OpenAI client — switch the model
dropdown in Open WebUI and you are A/B testing retrieval strategies against the
same corpus, with no bespoke UI.

Deliberate deviations from the OpenAI schema, all additive so clients that do
not know about them are unaffected:

  * `citations` on the response carries resolved passages with their page/bbox
    or character range. Unknown fields are ignored by every client tested.
  * `temperature`, `top_p`, `n`, `presence_penalty` and friends are accepted and
    ignored. Silently accepting them keeps clients working; pretending to honour
    them would be worse than not offering them.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ragent.agent.answer import answer_stream
from ragent.config import get_settings
from ragent.ingest.chunking import STRATEGIES
from ragent.providers import llm
from ragent.retrieval.search import Passage, hybrid_search

router = APIRouter(prefix="/v1", tags=["openai-compatible"])

MODEL_PREFIX = "ragent"
OWNER = "ragent"


def _strategy_for(model: str) -> str:
    """Map a requested model name onto a chunking strategy."""
    settings = get_settings()
    default = settings.chunk_strategies[0]

    if model in (MODEL_PREFIX, "", None):
        return default
    if model.startswith(f"{MODEL_PREFIX}-"):
        suffix = model[len(MODEL_PREFIX) + 1 :]
        if suffix in STRATEGIES:
            return suffix
        raise HTTPException(
            404,
            f"model {model!r} not found; expected one of "
            f"{[MODEL_PREFIX] + [f'{MODEL_PREFIX}-{s}' for s in sorted(STRATEGIES)]}",
        )
    # An unknown name is likelier a misconfigured client than a real request for
    # a passthrough model, and answering it would silently ignore the corpus.
    raise HTTPException(404, f"model {model!r} not found; this server only serves {MODEL_PREFIX}*")


# ---------------------------------------------------------------- schema


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None

    def as_text(self) -> str:
        """Flatten OpenAI's multipart content blocks down to text."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "\n".join(
                part.get("text", "") for part in self.content if part.get("type") == "text"
            )
        return ""


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_PREFIX
    messages: list[ChatMessage]
    stream: bool = False
    # Accepted for compatibility, deliberately unused — see the module docstring.
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    user: str | None = None
    #: RAGent extension: restrict retrieval to specific documents.
    document_ids: list[str] | None = Field(default=None)

    # Tolerate client-specific fields rather than 422-ing over one we ignore.
    model_config = ConfigDict(extra="allow")


def _query_of(messages: list[ChatMessage]) -> str:
    """The last user turn is the question.

    Prior turns are intentionally not used as retrieval context yet: naively
    concatenating a conversation into one query reliably degrades retrieval,
    and doing it properly means query rewriting, which belongs in the agent
    plane rather than here.
    """
    for message in reversed(messages):
        if message.role == "user":
            text = message.as_text().strip()
            if text:
                return text
    raise HTTPException(400, "no user message with text content")


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _citation_payload(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim resolved citations to what a client can act on."""
    return [
        {
            "marker": c["marker"],
            "document_id": c["document_id"],
            "document_name": c["document_name"],
            "provenance": c["provenance"],
            "section_path": c["section_path"],
            "regions": c["regions"],
            "char_start": c["char_start"],
            "char_end": c["char_end"],
            "text": c["text"],
        }
        for c in citations
    ]


# ---------------------------------------------------------------- models


@router.get("/models")
async def list_models() -> dict[str, Any]:
    created = int(time.time())
    names = [MODEL_PREFIX] + [f"{MODEL_PREFIX}-{s}" for s in sorted(STRATEGIES)]
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": created, "owned_by": OWNER} for name in names
        ],
    }


@router.get("/models/{model_id:path}")
async def retrieve_model(model_id: str) -> dict[str, Any]:
    _strategy_for(model_id)  # raises 404 for anything we do not serve
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": OWNER,
    }


# ---------------------------------------------------------------- chat


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Any:
    strategy = _strategy_for(request.model)
    query = _query_of(request.messages)

    passages = await hybrid_search(query, strategy=strategy, document_ids=request.document_ids)

    if request.stream:
        return StreamingResponse(
            _stream(request, query, passages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )
    return await _complete(request, query, passages)


async def _complete(
    request: ChatCompletionRequest, query: str, passages: list[Passage]
) -> dict[str, Any]:
    chunks: list[str] = []
    citations: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}

    async for chunk in answer_stream(query, passages):
        if chunk.type == "delta":
            chunks.append(chunk.text)
        elif chunk.type == "citations":
            citations = chunk.data or []
        elif chunk.type == "usage":
            usage = chunk.data or {}

    text = "".join(chunks)
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
        # RAGent extension. Clients that do not know this field ignore it.
        "citations": _citation_payload(citations),
    }


async def _stream(
    request: ChatCompletionRequest, query: str, passages: list[Passage]
) -> AsyncIterator[str]:
    """Emit OpenAI's streaming chunk format.

    Framed by hand with plain "\\n\\n" rather than through an SSE library: this
    is the one endpoint whose wire format is dictated by other people's clients,
    and CRLF framing has already cost this project a debugging session once.
    """
    completion_id = _completion_id()
    created = int(time.time())

    def frame(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            **extra,
        }
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    yield frame({"role": "assistant", "content": ""})

    citations: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}

    async for chunk in answer_stream(query, passages):
        if chunk.type == "delta" and chunk.text:
            yield frame({"content": chunk.text})
        elif chunk.type == "citations":
            citations = chunk.data or []
        elif chunk.type == "usage":
            usage = chunk.data or {}

    # Citations ride on the final chunk so a client reading only `delta.content`
    # sees a well-formed stream, while one that knows about them gets the data.
    yield frame(
        {},
        finish="stop",
        citations=_citation_payload(citations),
        usage={
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    )
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------- embeddings


class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str = "ragent-embeddings"
    encoding_format: Literal["float", "base64"] = "float"


@router.post("/embeddings")
async def embeddings(request: EmbeddingsRequest) -> dict[str, Any]:
    """Expose whatever embedder RAGent is configured with.

    Useful for checking that a client and the index agree on the model before
    diagnosing bad retrieval as a ranking problem.
    """
    if request.encoding_format != "float":
        raise HTTPException(400, "only encoding_format='float' is supported")

    from ragent.providers.embeddings import get_embedder

    texts = [request.input] if isinstance(request.input, str) else request.input
    embedder = get_embedder()
    vectors = await embedder.embed_documents(texts)

    return {
        "object": "list",
        "model": embedder.model,
        "data": [
            {"object": "embedding", "index": i, "embedding": vector}
            for i, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@router.get("/health")
async def v1_health() -> dict[str, Any]:
    """Some OpenAI-compatible clients probe /v1/health before connecting."""
    return {"status": "ok", "generation": llm.describe()}
