"""Embedding providers.

`local` is the default and it matters more than it looks: it means `make up &&
make seed` works for someone who just cloned the repo and has no API key. A demo
that needs a paid account before it does anything is a demo most reviewers never
see.

`openai` reaches any OpenAI-compatible `/embeddings` endpoint — OpenAI, Azure,
Ollama (`nomic-embed-text`, `mxbai-embed-large`), vLLM, LM Studio — differing
only in `OPENAI_BASE_URL`.

Dimensions are *discovered*, never assumed. A hardcoded table would be wrong the
moment someone points `OPENAI_BASE_URL` at a local server running a model this
file has never heard of, and a silent dimension mismatch surfaces much later as
an unhelpful Qdrant error.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Protocol

from ragent.config import get_settings

__all__ = ["Embedder", "get_embedder"]


class Embedder(Protocol):
    model: str
    provider: str

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    """fastembed, running in-process on CPU. No network, no key."""

    provider = "local"
    model = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=self.model)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # fastembed is synchronous and CPU-bound; off the event loop it goes,
        # otherwise one embed call stalls every other consumer on this worker.
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


class VoyageEmbedder:
    provider = "voyage"
    model = "voyage-3"

    def __init__(self, api_key: str) -> None:
        import voyageai

        self._client = voyageai.AsyncClient(api_key=api_key)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # input_type matters: Voyage encodes documents and queries
        # asymmetrically, and mixing the two up quietly costs recall.
        result = await self._client.embed(texts, model=self.model, input_type="document")
        return list(result.embeddings)

    async def embed_query(self, text: str) -> list[float]:
        result = await self._client.embed([text], model=self.model, input_type="query")
        return list(result.embeddings[0])


class OpenAIEmbedder:
    """Any OpenAI-compatible /embeddings endpoint."""

    provider = "openai"

    #: Requests are chunked because hosted endpoints cap inputs per call and
    #: self-hosted ones fall over well before the documented limit.
    BATCH = 96

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        import openai

        self.model = model
        self._client = openai.AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH):
            batch = texts[start : start + self.BATCH]
            response = await self._client.embeddings.create(model=self.model, input=batch)
            # Some compatible servers return data out of order; index is
            # authoritative, position is not.
            ordered = sorted(response.data, key=lambda d: d.index)
            out.extend(list(d.embedding) for d in ordered)
        return out

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    settings = get_settings()
    backend = settings.embedding_backend.lower()

    if backend == "voyage":
        if not settings.voyage_api_key:
            raise RuntimeError(
                "EMBEDDING_BACKEND=voyage but VOYAGE_API_KEY is empty; "
                "set the key or switch back to EMBEDDING_BACKEND=local"
            )
        return VoyageEmbedder(settings.voyage_api_key)

    if backend == "openai":
        return OpenAIEmbedder(
            settings.openai_api_key,
            settings.openai_base_url,
            settings.openai_embedding_model,
        )

    if backend != "local":
        raise ValueError(
            f"unknown EMBEDDING_BACKEND {backend!r}; expected 'local', 'openai' or 'voyage'"
        )

    return LocalEmbedder()
