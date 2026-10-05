"""Leak test: run every gold document through the real gateway with a recording upstream, and count the
gold PII values that reach the upstream payload. This is the headline safety number.

Each document is sent twice: as a user message, and as a tool result (the user message then only asks
the model to act on the tool output), so the tool-call path is measured too. A value counts as leaked
when it appears verbatim in any string the upstream received; numbers also count
when their digits appear as one run (so "4111-1111..." vs "4111 1111..." formatting does not hide a
leak). Text values must match as whole words, and names, addresses and organizations case-sensitively, so
the name "Tom" is not counted as leaked by the word "customer". Person names additionally report a partial
leak: any name token of 3+ letters that survives (an upper bound).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import httpx
from pydantic import BaseModel, Field

from pii_shield.config import Settings
from pii_shield.detect.pipeline import DetectionPipeline
from pii_shield.entities import EntityType
from pii_shield.eval.gold import GoldDoc
from pii_shield.gateway.app import create_app
from pii_shield.gateway.runtime import build_runtime
from pii_shield.providers.fake import FakeProvider, RecordingProvider

_NUMERIC = {EntityType.CREDIT_CARD, EntityType.IBAN, EntityType.PHONE, EntityType.US_SSN, EntityType.NATIONAL_ID}
CHANNELS = ("user", "tool_result")


class TypeLeak(BaseModel):
    total: int = 0
    leaked: int = 0


class LeakExample(BaseModel):
    doc: str
    channel: str
    type: str
    value: str
    partial: bool = False


class LeakResult(BaseModel):
    policy: str
    detectors: str
    channel: str
    requests: int
    blocked_requests: int
    protected_values: int
    leaked: int
    leaked_partial_names: int
    leak_rate: float
    by_type: dict[str, TypeLeak] = Field(default_factory=dict)
    examples: list[LeakExample] = Field(default_factory=list)


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def _digit_runs(text: str) -> set[str]:
    return {re.sub(r"\D", "", run) for run in re.findall(r"\+?\d[\d\s().-]{4,}\d", text)}


_CASE_INSENSITIVE = {EntityType.EMAIL, EntityType.URL, EntityType.IBAN, EntityType.SECRET, EntityType.IP_ADDRESS}


def is_leaked(kind: EntityType, value: str, sent: str, sent_digits: set[str]) -> bool:
    """Whole-word occurrence of the gold value in what the upstream received.

    Names, addresses and organizations match case-sensitively ("Tom" is not leaked by "customer", the name
    "Ivy" not by the plant "ivy" left in the text); emails, URLs and codes match case-insensitively.
    """
    flags = re.IGNORECASE if kind in _CASE_INSENSITIVE else 0
    if re.search(rf"(?<![\w]){re.escape(value)}(?![\w])", sent, flags):
        return True
    if kind in _NUMERIC:
        digits = re.sub(r"\D", "", value)
        return len(digits) >= 6 and any(digits in run for run in sent_digits)
    return False


def partial_name_leak(value: str, sent: str) -> bool:
    """Any name token of 3+ letters as a whole word (an upper bound: "Will" also matches "Will the update...")."""
    tokens = re.findall(r"[^\W\d_]{3,}", value)
    return any(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", sent) for t in tokens)


def request_body(doc: GoldDoc, channel: str) -> dict[str, Any]:
    system = {"role": "system", "content": "You are a helpful assistant for Brightloop's support team."}
    if channel == "user":
        return {"model": "fake-echo", "messages": [system, {"role": "user", "content": doc.text}]}
    call = {
        "id": "call_lookup",
        "type": "function",
        "function": {"name": "lookup_ticket", "arguments": '{"ticket": "' + doc.id + '"}'},
    }
    return {
        "model": "fake-echo",
        "messages": [
            system,
            {"role": "user", "content": "Summarize the ticket returned by the tool."},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_lookup", "content": doc.text},
        ],
    }


async def run_leak_test(
    docs: list[GoldDoc],
    settings: Settings,
    pipeline: DetectionPipeline,
    policies: list[str],
    detectors_label: str,
    channels: tuple[str, ...] = CHANNELS,
) -> list[LeakResult]:
    recorder = RecordingProvider(FakeProvider())
    runtime = build_runtime(settings, upstream=recorder, pipeline=pipeline)
    app = create_app(settings, runtime)
    results: list[LeakResult] = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        for policy_name in policies:
            policy = runtime.shield.policies.get(policy_name)
            protected = policy.protected_types
            for channel in channels:
                result = LeakResult(
                    policy=policy_name,
                    detectors=detectors_label,
                    channel=channel,
                    requests=0,
                    blocked_requests=0,
                    protected_values=0,
                    leaked=0,
                    leaked_partial_names=0,
                    leak_rate=0.0,
                )
                for doc in docs:
                    before = len(recorder.requests)
                    response = await client.post(
                        f"/p/{policy_name}/v1/chat/completions", json=request_body(doc, channel)
                    )
                    result.requests += 1
                    if response.status_code == 400 and response.json().get("error", {}).get("code") == "pii_blocked":
                        result.blocked_requests += 1
                    elif response.status_code != 200:
                        raise RuntimeError(
                            f"gateway returned {response.status_code} for {doc.id}: {response.text[:200]}"
                        )
                    sent = "\n".join(
                        s for body in recorder.requests[before:] for s in _strings(body.get("messages", []))
                    )
                    digits = _digit_runs(sent)
                    for entity in doc.entities:
                        if entity.type not in protected:
                            continue
                        stats = result.by_type.setdefault(entity.type.value, TypeLeak())
                        stats.total += 1
                        result.protected_values += 1
                        if is_leaked(entity.type, entity.value, sent, digits):
                            stats.leaked += 1
                            result.leaked += 1
                            result.examples.append(
                                LeakExample(doc=doc.id, channel=channel, type=entity.type.value, value=entity.value)
                            )
                        elif entity.type is EntityType.PERSON and partial_name_leak(entity.value, sent):
                            result.leaked_partial_names += 1
                            result.examples.append(
                                LeakExample(
                                    doc=doc.id,
                                    channel=channel,
                                    type=entity.type.value,
                                    value=entity.value,
                                    partial=True,
                                )
                            )
                result.leak_rate = result.leaked / result.protected_values if result.protected_values else 0.0
                result.by_type = dict(sorted(result.by_type.items()))
                results.append(result)
    return results
