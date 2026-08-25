"""Model routing and cost accounting.

Routing is by *task tier*, not by a dropdown. Grading a passage for relevance and
synthesising a cited answer are different jobs with different price/quality
curves, and sending both to the same model is either wasteful or bad. Each tier
names a model, and every call records what it actually cost so the routing
decisions are auditable after the fact rather than asserted in a README.

The API key is optional throughout. Without one the app still runs — retrieval
works, answers fall back to extractive passages — because a demo that needs a
paid account before it does anything is a demo most reviewers never see.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from ragent.config import get_settings

__all__ = ["Tier", "Usage", "LLMUnavailable", "complete", "stream_text", "available"]

log = logging.getLogger(__name__)


class Tier(StrEnum):
    #: Cited answer synthesis. Correctness matters more than cost here.
    SYNTHESIS = "synthesis"
    #: Relevance grading, query rewriting, chunk contextualisation. High volume.
    UTILITY = "utility"
    #: Figure and chart captioning.
    VISION = "vision"


#: USD per million tokens, (input, output). Used for the cost column in the
#: control-plane dashboard; update alongside any model change.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


class LLMUnavailable(RuntimeError):
    """No API key configured. Callers fall back rather than fail."""


@dataclass(frozen=True, slots=True)
class Usage:
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int

    @property
    def cost_usd(self) -> float:
        rate_in, rate_out = PRICING.get(self.model, (0.0, 0.0))
        return (self.input_tokens * rate_in + self.output_tokens * rate_out) / 1_000_000


def model_for(tier: Tier) -> str:
    settings = get_settings()
    return {
        Tier.SYNTHESIS: settings.model_synthesis,
        Tier.UTILITY: settings.model_utility,
        Tier.VISION: settings.model_vision,
    }[tier]


def available() -> bool:
    return bool(get_settings().anthropic_api_key)


@lru_cache(maxsize=1)
def _client():  # type: ignore[no-untyped-def]
    import anthropic

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


async def complete(
    prompt: str,
    *,
    tier: Tier = Tier.UTILITY,
    system: str | None = None,
    max_tokens: int = 4096,
    thinking: bool = False,
) -> tuple[str, Usage]:
    """One non-streaming call. Returns the text and what it cost."""
    import anthropic

    client = _client()
    model = model_for(tier)
    started = time.monotonic()

    kwargs: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    if thinking:
        # Adaptive thinking: the model decides how much reasoning a given
        # request needs, rather than us guessing a fixed budget.
        kwargs["thinking"] = {"type": "adaptive"}

    try:
        response = await client.messages.create(**kwargs)  # type: ignore[arg-type]
    except anthropic.RateLimitError:
        # Left to the caller's retry policy: the stage runner already knows how
        # to back this off, and rate limits do clear.
        raise
    except anthropic.APIStatusError as exc:
        if exc.status_code >= 500:
            raise
        raise LLMUnavailable(f"request rejected: {exc.message}") from exc

    text = "".join(block.text for block in response.content if block.type == "text")
    usage = Usage(
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=int((time.monotonic() - started) * 1000),
    )
    return text, usage


async def stream_text(
    prompt: str,
    *,
    tier: Tier = Tier.SYNTHESIS,
    system: str | None = None,
    max_tokens: int = 8192,
) -> AsyncIterator[tuple[str, Usage | None]]:
    """Yield (delta, None) as text arrives, then ('', usage) once at the end.

    Streaming is not cosmetic here: an answer over a dozen retrieved passages
    takes long enough that a non-streaming call reads as a hang, and large
    `max_tokens` values risk the SDK's HTTP timeout.
    """
    client = _client()
    model = model_for(tier)
    started = time.monotonic()

    kwargs: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    async with client.messages.stream(**kwargs) as stream:  # type: ignore[arg-type]
        async for delta in stream.text_stream:
            yield delta, None
        final = await stream.get_final_message()

    yield (
        "",
        Usage(
            model=model,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            latency_ms=int((time.monotonic() - started) * 1000),
        ),
    )
