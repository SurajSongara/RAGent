"""OpenAI compatibility, in both directions.

Outbound: RAGent can be driven by any OpenAI-compatible endpoint.
Inbound: RAGent can be consumed as one.

The wire-format tests matter more than they look. This is the one surface whose
shape is dictated by other people's clients — a stray framing difference does not
raise, it just makes Open WebUI or the openai SDK hang, and that is expensive to
diagnose from the outside.
"""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from ragent.api.openai_compat import (
    MODEL_PREFIX,
    ChatCompletionRequest,
    ChatMessage,
    _query_of,
    _strategy_for,
    _stream,
)
from ragent.config import Settings, get_settings
from ragent.providers import llm


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch):
    """Keep the developer's real .env out of these assertions."""
    get_settings.cache_clear()
    llm._provider.cache_clear()
    yield
    get_settings.cache_clear()
    llm._provider.cache_clear()


def settings_with(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


# ---------------------------------------------------------------- provider choice


class TestProviderSelection:
    def _use(self, monkeypatch, **kwargs) -> None:
        monkeypatch.setattr(llm, "get_settings", lambda: settings_with(**kwargs))
        llm._provider.cache_clear()

    def test_no_credentials_means_no_provider(self, monkeypatch) -> None:
        """Not an error — the answer path falls back to extractive passages."""
        self._use(monkeypatch)
        assert llm.available() is False
        assert llm.active_provider() is None
        assert llm.describe()["extractive_fallback"] is True

    def test_auto_prefers_anthropic(self, monkeypatch) -> None:
        self._use(monkeypatch, anthropic_api_key="sk-ant-x", openai_api_key="sk-x")
        assert llm.active_provider() == "anthropic"

    def test_auto_falls_back_to_openai(self, monkeypatch) -> None:
        self._use(monkeypatch, openai_api_key="sk-x")
        assert llm.active_provider() == "openai"

    def test_auto_detects_a_local_endpoint_without_a_key(self, monkeypatch) -> None:
        """Ollama and vLLM need no key; a custom base URL is the only signal."""
        self._use(monkeypatch, openai_base_url="http://localhost:11434/v1")
        assert llm.active_provider() == "openai"

    def test_explicit_openai_wins_over_an_anthropic_key(self, monkeypatch) -> None:
        self._use(monkeypatch, llm_provider="openai", anthropic_api_key="sk-ant-x")
        assert llm.active_provider() == "openai"

    def test_none_forces_the_fallback(self, monkeypatch) -> None:
        """What the eval harness uses to measure retrieval without generation."""
        self._use(monkeypatch, llm_provider="none", anthropic_api_key="sk-ant-x")
        assert llm.available() is False

    def test_explicit_anthropic_without_a_key_is_an_error(self, monkeypatch) -> None:
        self._use(monkeypatch, llm_provider="anthropic")
        assert llm.available() is False

    def test_unknown_provider_is_rejected_at_config_time(self) -> None:
        with pytest.raises(ValueError, match="LLM_PROVIDER must be one of"):
            settings_with(llm_provider="bedrock")

    def test_base_url_trailing_slash_is_stripped(self) -> None:
        """The SDK appends "/chat/completions"; a trailing slash yields "//"."""
        assert settings_with(openai_base_url="http://x/v1/").openai_base_url == "http://x/v1"


class TestUsageAccounting:
    def test_known_model_is_priced(self) -> None:
        usage = llm.Usage("gpt-4o-mini", "openai", 1_000_000, 1_000_000, 10)
        assert usage.priced is True
        assert usage.cost_usd == pytest.approx(0.15 + 0.60)

    def test_unknown_model_reports_unpriced_rather_than_guessing(self) -> None:
        """A self-hosted model has no list price; inventing one would mislead."""
        usage = llm.Usage("llama3.2:3b", "openai", 1_000_000, 1_000_000, 10)
        assert usage.priced is False
        assert usage.cost_usd == 0.0


# ---------------------------------------------------------------- model routing


class TestModelNameRouting:
    def test_bare_name_uses_the_configured_default(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ragent.api.openai_compat.get_settings",
            lambda: settings_with(chunk_strategies=["recursive"]),
        )
        assert _strategy_for(MODEL_PREFIX) == "recursive"

    @pytest.mark.parametrize("strategy", ["layout", "recursive", "fixed", "semantic"])
    def test_suffix_selects_a_chunking_strategy(self, strategy: str) -> None:
        """This is what makes the bake-off drivable from any OpenAI client."""
        assert _strategy_for(f"{MODEL_PREFIX}-{strategy}") == strategy

    def test_unknown_suffix_is_a_404(self) -> None:
        with pytest.raises(HTTPException) as exc:
            _strategy_for(f"{MODEL_PREFIX}-magic")
        assert exc.value.status_code == 404

    def test_foreign_model_is_a_404(self) -> None:
        """Answering as gpt-4o would silently ignore the corpus."""
        with pytest.raises(HTTPException) as exc:
            _strategy_for("gpt-4o")
        assert exc.value.status_code == 404


class TestMessageHandling:
    def test_last_user_turn_is_the_query(self) -> None:
        messages = [
            ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="answer"),
            ChatMessage(role="user", content="second"),
        ]
        assert _query_of(messages) == "second"

    def test_system_prompts_are_not_treated_as_the_query(self) -> None:
        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="real question"),
        ]
        assert _query_of(messages) == "real question"

    def test_multipart_content_is_flattened(self) -> None:
        """Open WebUI and the SDK both send content as an array of parts."""
        message = ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "what is"},
                {"type": "image_url", "image_url": {"url": "..."}},
                {"type": "text", "text": "this"},
            ],
        )
        assert message.as_text() == "what is\nthis"

    def test_no_usable_user_message_is_a_400(self) -> None:
        with pytest.raises(HTTPException) as exc:
            _query_of([ChatMessage(role="system", content="hi")])
        assert exc.value.status_code == 400

    def test_unknown_client_fields_are_tolerated(self) -> None:
        """A 422 here would break clients over a field we simply do not use."""
        request = ChatCompletionRequest.model_validate(
            {
                "model": "ragent",
                "messages": [{"role": "user", "content": "hi"}],
                "frequency_penalty": 0.5,
                "logit_bias": {},
                "seed": 42,
            }
        )
        assert request.model == "ragent"

    def test_sampling_parameters_are_accepted_and_ignored(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.9}
        )
        assert request.temperature == 0.9  # recorded, never forwarded


# ---------------------------------------------------------------- wire format


async def collect_stream(model: str = "ragent") -> list[str]:
    request = ChatCompletionRequest(
        model=model, messages=[ChatMessage(role="user", content="q")], stream=True
    )
    # No passages: the deterministic "nothing relevant" path, which still has to
    # produce a correctly framed stream.
    return [frame async for frame in _stream(request, "q", [])]


class TestStreamingWireFormat:
    async def test_every_frame_is_a_data_line(self) -> None:
        for frame in await collect_stream():
            assert frame.startswith("data: ")
            assert frame.endswith("\n\n"), "frames must be blank-line terminated"

    async def test_terminates_with_the_done_sentinel(self) -> None:
        frames = await collect_stream()
        assert frames[-1] == "data: [DONE]\n\n"

    async def test_first_chunk_announces_the_role(self) -> None:
        payload = json.loads((await collect_stream())[0].removeprefix("data: "))
        assert payload["choices"][0]["delta"]["role"] == "assistant"
        assert payload["object"] == "chat.completion.chunk"

    async def test_exactly_one_finish_reason(self) -> None:
        finishes = [
            json.loads(f.removeprefix("data: "))["choices"][0]["finish_reason"]
            for f in await collect_stream()
            if f != "data: [DONE]\n\n"
        ]
        assert finishes.count("stop") == 1
        assert finishes[-1] == "stop"

    async def test_completion_id_is_stable_across_the_stream(self) -> None:
        ids = {
            json.loads(f.removeprefix("data: "))["id"]
            for f in await collect_stream()
            if f != "data: [DONE]\n\n"
        }
        assert len(ids) == 1

    async def test_echoes_the_requested_model(self) -> None:
        payload = json.loads((await collect_stream("ragent-fixed"))[0].removeprefix("data: "))
        assert payload["model"] == "ragent-fixed"

    async def test_no_crlf_anywhere(self) -> None:
        """CRLF framing has already cost this project one debugging session."""
        assert not any("\r" in frame for frame in await collect_stream())
