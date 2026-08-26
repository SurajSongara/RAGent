"""Check the configured model provider end to end.

    python -m scripts.check_provider

Answers, in order: which provider is selected and why, what models the endpoint
actually offers, whether a completion works, whether streaming works, and
whether embeddings work. Run it after changing `.env` and before wondering why
answers look wrong.

It exists because "the model name is wrong" and "the key is wrong" and "this
endpoint has no embeddings" all present identically once they are buried inside
an ingest stage: the document just fails. Here they are three distinct lines.

The API key is never printed.
"""

from __future__ import annotations

import asyncio
import sys

from ragent.config import get_settings
from ragent.providers import llm


def mask(secret: str) -> str:
    if not secret:
        return "<empty>"
    return f"{secret[:6]}...{secret[-4:]} ({len(secret)} chars)"


async def list_models(base_url: str, api_key: str) -> list[str]:
    import openai

    client = openai.AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)
    page = await client.models.list()
    return sorted(m.id for m in page.data)


async def main() -> int:
    settings = get_settings()
    failures = 0

    print("=== configuration ===")
    print(f"  LLM_PROVIDER      {settings.llm_provider}")
    print(f"  ANTHROPIC_API_KEY {mask(settings.anthropic_api_key)}")
    print(f"  OPENAI_API_KEY    {mask(settings.openai_api_key)}")
    print(f"  OPENAI_BASE_URL   {settings.openai_base_url}")
    print(f"  EMBEDDING_BACKEND {settings.embedding_backend}")

    print("\n=== resolved provider ===")
    provider = llm.active_provider()
    if provider is None:
        print("  none - answers fall back to ranked extractive passages.")
        print("  Set ANTHROPIC_API_KEY, or OPENAI_API_KEY (+ OPENAI_BASE_URL).")
        return 1
    print(f"  provider {provider}")
    for tier, model in llm.describe()["models"].items():
        print(f"  {tier:10s} {model}")

    if provider == "openai":
        print("\n=== models offered by the endpoint ===")
        try:
            models = await list_models(settings.openai_base_url, settings.openai_api_key)
            for name in models:
                print(f"  {name}")
            wanted = settings.openai_model_synthesis
            if models and wanted not in models:
                # The single most common misconfiguration, and the error it
                # produces deep in a stage handler says nothing useful.
                print(
                    f"\n  WARNING: OPENAI_MODEL_SYNTHESIS={wanted!r} is not in that list. "
                    "Pick one of the names above."
                )
                failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  could not list models: {exc}")
            failures += 1

    print("\n=== completion ===")
    try:
        text, usage = await llm.complete(
            "Reply with exactly: ok", tier=llm.Tier.SYNTHESIS, max_tokens=32
        )
        print(f"  reply  {text.strip()[:80]!r}")
        price = f"${usage.cost_usd:.6f}" if usage.priced else "unpriced"
        print(
            f"  usage  {usage.model} in={usage.input_tokens} out={usage.output_tokens} "
            f"{price} {usage.latency_ms}ms"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        failures += 1

    print("\n=== streaming ===")
    try:
        parts: list[str] = []
        async for delta, _ in llm.stream_text(
            "Count: one two three", tier=llm.Tier.SYNTHESIS, max_tokens=32
        ):
            if delta:
                parts.append(delta)
        print(f"  received {len(parts)} deltas: {''.join(parts).strip()[:80]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        failures += 1

    print("\n=== embeddings ===")
    try:
        from ragent.providers.embeddings import get_embedder

        embedder = get_embedder()
        vector = await embedder.embed_query("hello")
        print(f"  {embedder.provider}/{embedder.model} -> {len(vector)} dims")
    except Exception as exc:  # noqa: BLE001
        # Several chat providers, xAI among them, serve no /embeddings route.
        # That is fine: EMBEDDING_BACKEND=local is independent of the chat model.
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        print("  If this endpoint has no embeddings API, keep EMBEDDING_BACKEND=local.")
        failures += 1

    print("\n" + ("all checks passed" if not failures else f"{failures} check(s) failed"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
