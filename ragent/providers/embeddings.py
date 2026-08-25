"""Embedding providers.

`local` is the default and it matters more than it looks: it means `make up &&
make seed` works for someone who just cloned the repo and has no API keys. A
demo that needs a paid account before it does anything is a demo most reviewers
never see.

`voyage` is the hosted upgrade for when retrieval quality is being measured
rather than demonstrated.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Protocol

from ragent.config import get_settings

__all__ = ["Embedder", "get_embedder", "EMBEDDING_DIMS"]

#: Qdrant collections are created with a fixed vector size, so changing the
#: model means reindexing. Recorded here so the mismatch fails loudly.
EMBEDDING_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "voyage-3": 1024,
}


class Embedder(Protocol):
    model: str
    dims: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    """fastembed, running in-process on CPU. No network, no key."""

    model = "BAAI/bge-small-en-v1.5"
    dims = EMBEDDING_DIMS["BAAI/bge-small-en-v1.5"]

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
    model = "voyage-3"
    dims = EMBEDDING_DIMS["voyage-3"]

    def __init__(self, api_key: str) -> None:
        import voyageai

        self._client = voyageai.AsyncClient(api_key=api_key)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # input_type matters: Voyage encodes documents and queries asymmetrically,
        # and mixing the two up quietly costs recall.
        result = await self._client.embed(texts, model=self.model, input_type="document")
        return list(result.embeddings)

    async def embed_query(self, text: str) -> list[float]:
        result = await self._client.embed([text], model=self.model, input_type="query")
        return list(result.embeddings[0])


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

    if backend != "local":
        raise ValueError(f"unknown EMBEDDING_BACKEND {backend!r}; expected 'local' or 'voyage'")

    return LocalEmbedder()
