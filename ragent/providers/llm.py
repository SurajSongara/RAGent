"""Model routing across providers, with cost accounting.

Two axes here, and they are independent.

**Which provider.** Anthropic and any OpenAI-compatible endpoint are both
first-class. "OpenAI-compatible" is doing real work in that sentence: the same
code path reaches OpenAI, Azure OpenAI, Ollama, vLLM, Groq, Together, OpenRouter
and LM Studio, because they all speak the same wire format and differ only in
`OPENAI_BASE_URL`. That means the project is not an advertisement for one vendor,
and a reviewer can run it against a local Ollama with no account at all.

**Which tier.** Grading a passage for relevance and synthesising a cited answer
are different jobs with different price/quality curves. Each tier names a model
per provider, and every call records what it actually cost, so the routing
decisions are auditable rather than asserted.

Having no key at all remains a supported configuration, not a broken one — the
answer path falls back to ranked extractive passages.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Any, Protocol

from ragent.config import get_settings

__all__ = [
    "Tier",
    "Usage",
    "LLMUnavailable",
    "complete",
    "stream_text",
    "available",
    "active_provider",
    "describe",
]

log = logging.getLogger(__name__)


class Tier(StrEnum):
    #: Cited answer synthesis. Correctness matters more than cost here.
    SYNTHESIS = "synthesis"
    #: Relevance grading, query rewriting, chunk contextualisation. High volume.
    UTILITY = "utility"
    #: Figure and chart captioning.
    VISION = "vision"


#: USD per million tokens, (input, output). Anything absent bills as 0 and is
#: reported as unpriced rather than guessed — a self-hosted Ollama model has no
#: list price, and inventing one would make the cost dashboard a liar.
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


class LLMUnavailable(RuntimeError):
    """No usable provider is configured. Callers fall back rather than fail."""


@dataclass(frozen=True, slots=True)
class Usage:
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: int

    @property
    def priced(self) -> bool:
        return self.model in PRICING

    @property
    def cost_usd(self) -> float:
        rate_in, rate_out = PRICING.get(self.model, (0.0, 0.0))
        return (self.input_tokens * rate_in + self.output_tokens * rate_out) / 1_000_000


class Provider(Protocol):
    name: str

    def model_for(self, tier: Tier) -> str: ...
    async def complete(
        self, prompt: str, *, tier: Tier, system: str | None, max_tokens: int, thinking: bool
    ) -> tuple[str, Usage]: ...
    def stream_text(
        self, prompt: str, *, tier: Tier, system: str | None, max_tokens: int
    ) -> AsyncIterator[tuple[str, Usage | None]]: ...


# ---------------------------------------------------------------- anthropic


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        import anthropic

        self._sdk = anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    def model_for(self, tier: Tier) -> str:
        s = get_settings()
        return {
            Tier.SYNTHESIS: s.model_synthesis,
            Tier.UTILITY: s.model_utility,
            Tier.VISION: s.model_vision,
        }[tier]

    async def complete(
        self, prompt: str, *, tier: Tier, system: str | None, max_tokens: int, thinking: bool
    ) -> tuple[str, Usage]:
        model = self.model_for(tier)
        started = time.monotonic()

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if thinking:
            # Adaptive: the model decides how much reasoning the request needs,
            # rather than us guessing a fixed budget.
            kwargs["thinking"] = {"type": "adaptive"}

        try:
            response = await self._client.messages.create(**kwargs)
        except self._sdk.RateLimitError:
            raise  # the stage runner's backoff already handles this
        except self._sdk.APIStatusError as exc:
            if exc.status_code >= 500:
                raise
            raise LLMUnavailable(f"request rejected: {exc.message}") from exc

        text = "".join(b.text for b in response.content if b.type == "text")
        return text, Usage(
            model=model,
            provider=self.name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    async def stream_text(
        self, prompt: str, *, tier: Tier, system: str | None, max_tokens: int
    ) -> AsyncIterator[tuple[str, Usage | None]]:
        model = self.model_for(tier)
        started = time.monotonic()

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for delta in stream.text_stream:
                yield delta, None
            final = await stream.get_final_message()

        yield (
            "",
            Usage(
                model=model,
                provider=self.name,
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                latency_ms=int((time.monotonic() - started) * 1000),
            ),
        )


# ---------------------------------------------------------------- openai


class OpenAIProvider:
    """Any endpoint speaking the OpenAI chat-completions wire format."""

    name = "openai"

    def __init__(self, api_key: str, base_url: str) -> None:
        import openai

        self._sdk = openai
        # Local servers (Ollama, vLLM, LM Studio) ignore the key but the SDK
        # refuses to construct without one, so a placeholder keeps them usable.
        self._client = openai.AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)

    def model_for(self, tier: Tier) -> str:
        s = get_settings()
        return {
            Tier.SYNTHESIS: s.openai_model_synthesis,
            Tier.UTILITY: s.openai_model_utility,
            Tier.VISION: s.openai_model_vision,
        }[tier]

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        return messages

    async def complete(
        self, prompt: str, *, tier: Tier, system: str | None, max_tokens: int, thinking: bool
    ) -> tuple[str, Usage]:
        model = self.model_for(tier)
        started = time.monotonic()

        try:
            response = await self._client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=self._messages(prompt, system),  # type: ignore[arg-type]
            )
        except self._sdk.RateLimitError:
            raise
        except self._sdk.APIStatusError as exc:
            if exc.status_code >= 500:
                raise
            raise LLMUnavailable(f"request rejected: {exc}") from exc

        usage = response.usage
        return (response.choices[0].message.content or ""), Usage(
            model=model,
            provider=self.name,
            # Self-hosted servers frequently omit usage entirely.
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    async def stream_text(
        self, prompt: str, *, tier: Tier, system: str | None, max_tokens: int
    ) -> AsyncIterator[tuple[str, Usage | None]]:
        model = self.model_for(tier)
        started = time.monotonic()

        stream = await self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=self._messages(prompt, system),  # type: ignore[arg-type]
            stream=True,
            # Not every compatible server honours this; when it is missing the
            # token counts come back as zero rather than the call failing.
            stream_options={"include_usage": True},
        )

        prompt_tokens = completion_tokens = 0
        async for chunk in stream:
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta, None

        yield (
            "",
            Usage(
                model=model,
                provider=self.name,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                latency_ms=int((time.monotonic() - started) * 1000),
            ),
        )


# ---------------------------------------------------------------- selection


@lru_cache(maxsize=1)
def _provider() -> Provider | None:
    s = get_settings()
    choice = s.llm_provider

    if choice == "none":
        return None
    if choice == "anthropic":
        if not s.anthropic_api_key:
            raise LLMUnavailable("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty")
        return AnthropicProvider(s.anthropic_api_key)
    if choice == "openai":
        # A local endpoint needs no key, so the base URL alone is enough.
        return OpenAIProvider(s.openai_api_key, s.openai_base_url)

    # auto: prefer Anthropic, fall back to any configured OpenAI-compatible
    # endpoint, then to no provider at all.
    if s.anthropic_api_key:
        return AnthropicProvider(s.anthropic_api_key)
    if s.openai_api_key or s.openai_base_url != "https://api.openai.com/v1":
        return OpenAIProvider(s.openai_api_key, s.openai_base_url)
    return None


def available() -> bool:
    try:
        return _provider() is not None
    except LLMUnavailable:
        return False


def active_provider() -> str | None:
    provider = _provider()
    return provider.name if provider else None


def describe() -> dict[str, Any]:
    """What the /health endpoint reports about generation."""
    provider = _provider()
    if provider is None:
        return {"provider": None, "models": {}, "extractive_fallback": True}
    return {
        "provider": provider.name,
        "models": {str(t): provider.model_for(t) for t in Tier},
        "base_url": get_settings().openai_base_url if provider.name == "openai" else None,
        "extractive_fallback": False,
    }


def _require() -> Provider:
    provider = _provider()
    if provider is None:
        raise LLMUnavailable(
            "no model provider configured; set ANTHROPIC_API_KEY, or OPENAI_API_KEY "
            "(optionally with OPENAI_BASE_URL for Ollama/vLLM/Groq/OpenRouter)"
        )
    return provider


async def complete(
    prompt: str,
    *,
    tier: Tier = Tier.UTILITY,
    system: str | None = None,
    max_tokens: int = 4096,
    thinking: bool = False,
) -> tuple[str, Usage]:
    return await _require().complete(
        prompt, tier=tier, system=system, max_tokens=max_tokens, thinking=thinking
    )


async def stream_text(
    prompt: str,
    *,
    tier: Tier = Tier.SYNTHESIS,
    system: str | None = None,
    max_tokens: int = 8192,
) -> AsyncIterator[tuple[str, Usage | None]]:
    """Yield (delta, None) as text arrives, then ('', usage) once at the end.

    Streaming is not cosmetic: an answer over a dozen retrieved passages takes
    long enough that a non-streaming call reads as a hang, and large `max_tokens`
    values risk the SDK's HTTP timeout.
    """
    async for item in _require().stream_text(
        prompt, tier=tier, system=system, max_tokens=max_tokens
    ):
        yield item
