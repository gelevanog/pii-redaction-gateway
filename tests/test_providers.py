"""Free-only guard, OpenAI/OpenRouter HTTP handling, retries/cache/budget, and the Anthropic conversion."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from pii_shield.config import Settings
from pii_shield.eval.config import EvalConfig, LlmModelConfig, UtilityConfig, check_paid_opt_in
from pii_shield.providers.anthropic_provider import from_anthropic_message, to_anthropic_request
from pii_shield.providers.base import (
    BudgetExceededError,
    FreeModelGuardError,
    ProviderError,
    RetryableError,
    ensure_free_models,
)
from pii_shield.providers.factory import build_provider
from pii_shield.providers.fake import FakeProvider
from pii_shield.providers.openai_compat import OpenAICompatibleProvider
from pii_shield.providers.resilient import CallLedger, DiskCache, ResilientProvider

FREE = "nvidia/nemotron-3-super-120b-a12b:free"


def completion(model: str = FREE, content: str = "hi", finish: str = "stop") -> dict[str, Any]:
    return {
        "id": "x",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }


def openrouter(handler: Any, **kwargs: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        kind="openrouter",
        api_key="test",
        base_url="https://openrouter.test/api/v1",
        default_model=kwargs.pop("model", FREE),
        require_free=kwargs.pop("require_free", True),
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def test_ensure_free_models() -> None:
    ensure_free_models([FREE, "qwen/qwen3.8-27b:free"])
    with pytest.raises(FreeModelGuardError):
        ensure_free_models([FREE, "openai/gpt-5.4-mini"])


def test_guard_on_construction_and_per_request() -> None:
    with pytest.raises(FreeModelGuardError):
        openrouter(lambda r: httpx.Response(200), model="anthropic/claude-sonnet-5")
    provider = openrouter(lambda r: httpx.Response(200), fallback_models=["qwen/qwen3.8-27b:free"])
    with pytest.raises(FreeModelGuardError):
        provider.prepare({"model": "openai/gpt-5.4", "messages": []})
    with pytest.raises(FreeModelGuardError):
        provider.prepare({"model": FREE, "models": [FREE, "openai/gpt-5.4"], "messages": []})
    prepared = provider.prepare({"model": "auto", "messages": []})
    assert prepared["model"] == FREE and prepared["models"] == [FREE, "qwen/qwen3.8-27b:free"]


def test_guard_can_be_disabled_for_paid_models() -> None:
    provider = openrouter(lambda r: httpx.Response(200), model="openai/gpt-5.4", require_free=False)
    assert provider.prepare({"messages": []})["model"] == "openai/gpt-5.4"


def test_eval_config_rejects_paid_models() -> None:
    with pytest.raises(ValueError, match="free"):
        EvalConfig(llm_detector=LlmModelConfig(model="openai/gpt-5.4", fallback_models=[]))
    with pytest.raises(ValueError, match="always paid"):
        EvalConfig(llm_detector=LlmModelConfig(provider="anthropic", model="claude-sonnet-5", fallback_models=[]))


def test_default_settings_do_not_allow_paid_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PII_SHIELD_ALLOW_PAID_MODELS", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.require_free_models and not settings.allow_paid_models
    default = EvalConfig()
    check_paid_opt_in(default, settings.allow_paid_models)  # free default config: nothing to opt into
    with pytest.raises(FreeModelGuardError):
        build_provider(
            settings.model_copy(update={"openrouter_api_key": "k"}), kind="openrouter", model="anthropic/claude-x"
        )


def test_paid_eval_config_needs_the_explicit_opt_in() -> None:
    paid = LlmModelConfig(
        provider="openai", model="claude-haiku-5-5", fallback_models=[], base_url="http://127.0.0.1:8787/v1"
    )
    config = EvalConfig(require_free_models=False, llm_detector=paid, utility=UtilityConfig(model=paid, judge=paid))
    with pytest.raises(ValueError, match="PII_SHIELD_ALLOW_PAID_MODELS"):
        check_paid_opt_in(config, allow_paid_models=False)
    check_paid_opt_in(config, allow_paid_models=True)
    with pytest.raises(ValueError, match="always paid"):  # the opt-in never weakens a free-only config
        EvalConfig(llm_detector=paid)


def test_opt_in_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PII_SHIELD_ALLOW_PAID_MODELS", "1")
    assert Settings(_env_file=None).allow_paid_models  # type: ignore[call-arg]


async def test_per_role_base_url_overrides_settings() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=completion(model="claude-haiku-5-5"))

    settings = Settings(_env_file=None, openai_api_key="test-key")  # type: ignore[call-arg]
    provider = build_provider(settings, kind="openai", model="claude-haiku-5-5", base_url="http://proxy.test/v1")
    assert isinstance(provider, OpenAICompatibleProvider)
    provider._transport = httpx.MockTransport(handler)
    answer = await provider.complete({"messages": [{"role": "user", "content": "x"}]})
    assert seen == ["http://proxy.test/v1/chat/completions"] and answer["model"] == "claude-haiku-5-5"


async def test_served_paid_model_is_refused() -> None:
    provider = openrouter(lambda r: httpx.Response(200, json=completion(model="openai/gpt-5.4")))
    with pytest.raises(FreeModelGuardError):
        await provider.complete({"messages": [{"role": "user", "content": "x"}]})


async def test_request_shape_and_auth_header() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion())

    answer = await openrouter(handler).complete(
        {"model": FREE, "messages": [{"role": "user", "content": "x"}], "custom_field": 1}
    )
    assert answer["choices"][0]["message"]["content"] == "hi"
    assert seen["auth"] == "Bearer test" and seen["body"]["custom_field"] == 1 and seen["body"]["stream"] is False


@pytest.mark.parametrize(
    "response,error",
    [
        (
            httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "rate-limited upstream"}}),
            RetryableError,
        ),
        (httpx.Response(200, json={"error": {"message": "upstream overloaded", "code": 502}}), RetryableError),
        (httpx.Response(401, json={"error": {"message": "bad key"}}), ProviderError),
        (httpx.Response(429, json={"error": {"message": "spent", "type": "budget_exhausted"}}), ProviderError),
        (httpx.Response(429, json={"error": {"message": "quota", "code": "insufficient_quota"}}), ProviderError),
        (httpx.Response(200, json=completion(content="", finish="length")), ProviderError),
        (httpx.Response(200, json=completion(content="", finish="stop")), RetryableError),
    ],
)
async def test_error_mapping(response: httpx.Response, error: type[Exception]) -> None:
    with pytest.raises(error) as excinfo:
        await openrouter(lambda r: response).complete({"messages": [{"role": "user", "content": "x"}]})
    if response.status_code == 429 and error is RetryableError:
        assert isinstance(excinfo.value, RetryableError) and excinfo.value.retry_after == 7.0
    if error is ProviderError:
        assert not isinstance(excinfo.value, RetryableError)


async def test_stream_parses_sse() -> None:
    chunks = [
        {"model": FREE, "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]}
        for c in ("Hel", "lo")
    ]
    body = ": OPENROUTER PROCESSING\n\n" + "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    provider = openrouter(lambda r: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"}))
    text = "".join([c["choices"][0]["delta"]["content"] async for c in provider.stream({"messages": []})])
    assert text == "Hello"


class Flaky:
    def __init__(self, failures: int, remote: bool = True) -> None:
        self.failures, self.calls, self.remote = failures, 0, remote

    @property
    def label(self) -> str:
        return "flaky/model:free"

    @property
    def is_remote(self) -> bool:
        return self.remote

    async def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        if self.calls <= self.failures:
            raise RetryableError("rate limited", retry_after=0.01)
        return completion()

    async def stream(self, body: dict[str, Any]):  # type: ignore[no-untyped-def]
        yield completion()


async def test_retries_then_cache_and_ledger(tmp_path: Path) -> None:
    inner = Flaky(failures=2)
    ledger = CallLedger(tmp_path / "calls.jsonl", max_calls=10)
    provider = ResilientProvider(
        inner, ledger=ledger, cache=DiskCache(tmp_path / "cache"), retry_base_seconds=0.01, tag="t"
    )
    body = {"model": FREE, "messages": [{"role": "user", "content": "x"}]}
    assert (await provider.complete(body))["choices"]
    assert inner.calls == 3 and ledger.calls == 3
    cached = await provider.complete(body)
    assert cached["_cached"] is True and inner.calls == 3
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["retryable_error", "retryable_error", "ok"]
    assert all("content" not in json.dumps(r) for r in rows)  # the ledger never stores prompts
    assert CallLedger(tmp_path / "calls.jsonl", max_calls=10).calls == 3  # budget survives restarts


async def test_budget_is_enforced(tmp_path: Path) -> None:
    provider = ResilientProvider(Flaky(failures=0), ledger=CallLedger(tmp_path / "c.jsonl", max_calls=1))
    await provider.complete({"messages": [{"role": "user", "content": "a"}]})
    with pytest.raises(BudgetExceededError):
        await provider.complete({"messages": [{"role": "user", "content": "b"}]})


async def test_local_provider_is_not_counted(tmp_path: Path) -> None:
    ledger = CallLedger(tmp_path / "c.jsonl", max_calls=0)
    assert await ResilientProvider(FakeProvider(), ledger=ledger).complete(
        {"messages": [{"role": "user", "content": "x"}]}
    )
    assert ledger.calls == 0


def test_anthropic_request_conversion() -> None:
    body = {
        "model": "gpt-whatever",
        "temperature": 0.2,
        "max_tokens": 500,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Mail <EMAIL_1>"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "send", "arguments": '{"to": "<EMAIL_1>"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sent"},
            {"role": "tool", "tool_call_id": "c2", "content": "also"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "send", "description": "d", "parameters": {"type": "object"}}}
        ],
        "tool_choice": "required",
    }
    request = to_anthropic_request(body, "claude-sonnet-5")
    assert request["model"] == "claude-sonnet-5" and request["system"] == "Be brief." and "temperature" not in request
    assert request["messages"][1]["content"][0] == {
        "type": "tool_use",
        "id": "c1",
        "name": "send",
        "input": {"to": "<EMAIL_1>"},
    }
    assert [b["tool_use_id"] for b in request["messages"][2]["content"]] == ["c1", "c2"]  # merged into one user turn
    assert request["tools"][0]["input_schema"] == {"type": "object"} and request["tool_choice"] == {"type": "any"}


def test_anthropic_response_conversion() -> None:
    message = SimpleNamespace(
        id="msg_1",
        model="claude-sonnet-5",
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="text", text="Sending."),
            SimpleNamespace(type="tool_use", id="tu_1", name="send", input={"to": "<EMAIL_1>"}),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    result = from_anthropic_message(message, 0)
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls" and choice["message"]["content"] == "Sending."
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"to": "<EMAIL_1>"}
