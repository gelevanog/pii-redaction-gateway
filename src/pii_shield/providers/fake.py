"""Deterministic offline upstreams: no keys, no network, same input -> same output.

`FakeProvider` is a polite echo bot. It greets the first person placeholder it sees, quotes the last
user message and lists the other placeholders, so restore is visibly exercised; when tools are offered
it calls the first tool with placeholder arguments. Streams are cut into 5-character chunks, which
splits most placeholders across chunks on purpose.

`RecordingProvider` wraps any provider and keeps every request body it forwards: the leak test checks
that body for gold PII values.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from pii_shield.providers.base import ChatProvider, JsonDict, message_text

_PLACEHOLDER = re.compile(r"<([A-Z_]+)_(\d+)>")
_ARG_HINTS = (("mail", "EMAIL"), ("phone", "PHONE"), ("name", "PERSON"), ("customer", "PERSON"), ("address", "ADDRESS"))
CHUNK_SIZE = 5


def _completion_id(body: JsonDict) -> str:
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return f"chatcmpl-fake-{digest[:12]}"


def _tool_arguments(tool: JsonDict, placeholders: list[str], summary: str) -> JsonDict:
    properties: dict[str, Any] = (tool.get("parameters") or {}).get("properties") or {}
    arguments: JsonDict = {}
    for name, schema in properties.items():
        if schema.get("type", "string") != "string":
            continue
        wanted = next((kind for hint, kind in _ARG_HINTS if hint in name.lower()), None)
        match = next((p for p in placeholders if wanted and p.startswith(f"<{wanted}_")), None)
        arguments[name] = match or summary
    return arguments


def fake_reply(body: JsonDict) -> JsonDict:
    """The assistant message the fake upstream answers with (text or a tool call)."""
    messages = body.get("messages", [])
    last = messages[-1] if messages else {}
    users = [m for m in messages if m.get("role") == "user"]
    user_text = message_text(users[-1]) if users else ""
    placeholders = list(dict.fromkeys(m.group(0) for m in _PLACEHOLDER.finditer(user_text)))
    summary = " ".join(user_text.split())
    summary = summary if len(summary) <= 240 else summary[:237].rstrip() + "..."

    response_format = body.get("response_format") or {}
    if response_format.get("type") in {"json_schema", "json_object"}:
        return {"role": "assistant", "content": json.dumps({"entities": []})}

    tools = [t["function"] for t in body.get("tools") or [] if t.get("type") == "function"]
    if tools and last.get("role") == "user" and body.get("tool_choice") != "none":
        tool = tools[0]
        call_id = "call_" + hashlib.sha256(user_text.encode()).hexdigest()[:16]
        arguments = json.dumps(_tool_arguments(tool, placeholders, summary), ensure_ascii=False)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": tool["name"], "arguments": arguments}}
            ],
        }
    if last.get("role") == "tool":
        return {"role": "assistant", "content": f"Done. The tool returned: {message_text(last)}"}

    person = next((p for p in placeholders if p.startswith("<PERSON_")), None)
    others = [p for p in placeholders if p != person]
    lines = [f"Hi {person or 'there'}, thanks for your message.", f'You wrote: "{summary}"']
    if others:
        lines.append("Details I will use for the follow-up: " + ", ".join(others) + ".")
    return {"role": "assistant", "content": "\n".join(lines)}


class FakeProvider:
    def __init__(self, model: str = "fake-echo") -> None:
        self.model = model

    @property
    def label(self) -> str:
        return f"fake/{self.model}"

    @property
    def is_remote(self) -> bool:
        return False

    async def complete(self, body: JsonDict) -> JsonDict:
        message = fake_reply(body)
        prompt_tokens = sum(len(message_text(m).split()) for m in body.get("messages", []))
        completion_tokens = len((message.get("content") or "").split())
        return {
            "id": _completion_id(body),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(body.get("model") or self.model),
            "choices": [
                {"index": 0, "message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def stream(self, body: JsonDict) -> AsyncIterator[JsonDict]:
        message = fake_reply(body)
        base = {
            "id": _completion_id(body),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": str(body.get("model") or self.model),
        }

        def chunk(delta: JsonDict, finish_reason: str | None = None) -> JsonDict:
            return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

        yield chunk({"role": "assistant", "content": ""})
        for index, call in enumerate(message.get("tool_calls") or []):
            head = {
                "index": index,
                "id": call["id"],
                "type": "function",
                "function": {"name": call["function"]["name"], "arguments": ""},
            }
            yield chunk({"tool_calls": [head]})
            arguments = call["function"]["arguments"]
            for start in range(0, len(arguments), CHUNK_SIZE):
                piece = {"index": index, "function": {"arguments": arguments[start : start + CHUNK_SIZE]}}
                yield chunk({"tool_calls": [piece]})
        content = message.get("content") or ""
        for start in range(0, len(content), CHUNK_SIZE):
            yield chunk({"content": content[start : start + CHUNK_SIZE]})
        yield chunk({}, "tool_calls" if message.get("tool_calls") else "stop")


class RecordingProvider:
    """Records every body sent upstream (the leak test's "network tap")."""

    def __init__(self, inner: ChatProvider) -> None:
        self.inner = inner
        self.requests: list[JsonDict] = []
        self.responses: list[JsonDict] = []

    @property
    def label(self) -> str:
        return self.inner.label

    @property
    def is_remote(self) -> bool:
        return self.inner.is_remote

    async def complete(self, body: JsonDict) -> JsonDict:
        self.requests.append(json.loads(json.dumps(body)))
        response = await self.inner.complete(body)
        self.responses.append(json.loads(json.dumps(response)))
        return response

    async def stream(self, body: JsonDict) -> AsyncIterator[JsonDict]:
        self.requests.append(json.loads(json.dumps(body)))
        async for chunk in self.inner.stream(body):
            yield chunk
