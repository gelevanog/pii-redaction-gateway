"""Redact OpenAI chat.completions requests and restore responses, including streamed chunks.

What gets redacted: every message's text content (system, developer, user, assistant, tool), text parts
of multi-part content, and the string values inside tool-call arguments (parsed as JSON so the arguments
stay valid JSON). The optional `name` field of a message is dropped: it may carry a real name and has
no redacted form that fits OpenAI's `[a-zA-Z0-9_-]` constraint.

What gets restored: answer text and tool-call arguments. Restoring arguments matters: when the model
calls `send_email(to="<EMAIL_1>")`, the client's tool must receive the real address.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pii_shield.anonymize.restore import RestoreReport, StreamRestorer
from pii_shield.providers.base import JsonDict
from pii_shield.shield import RedactionResult, ShieldSession

_NAME_ROLES = {"system", "developer", "user", "assistant"}


def _map_strings(value: Any, transform: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, list):
        return [_map_strings(item, transform) for item in value]
    if isinstance(value, dict):
        return {key: _map_strings(item, transform) for key, item in value.items()}
    return value


def _redact_arguments(arguments: str, redact: Callable[[str], str]) -> str:
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return redact(arguments)
    return json.dumps(_map_strings(parsed, redact), ensure_ascii=False)


def redact_chat_request(session: ShieldSession, body: JsonDict) -> tuple[JsonDict, list[RedactionResult]]:
    """A deep copy of `body` with every message redacted through one session (consistent placeholders)."""
    redacted = copy.deepcopy(body)
    results: list[RedactionResult] = []

    def redact(text: str) -> str:
        if not text.strip():
            return text
        result = session.redact(text)
        results.append(result)
        return result.text

    for message in redacted.get("messages", []):
        if message.get("role") in _NAME_ROLES:
            message.pop("name", None)
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = redact(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    part["text"] = redact(part["text"])
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if isinstance(function.get("arguments"), str):
                function["arguments"] = _redact_arguments(function["arguments"], redact)
        legacy = message.get("function_call")
        if isinstance(legacy, dict) and isinstance(legacy.get("arguments"), str):
            legacy["arguments"] = _redact_arguments(legacy["arguments"], redact)
    return redacted, results


PLACEHOLDER_HINT = (
    "Some personal data in this conversation was replaced by placeholders such as <PERSON_1> or <EMAIL_1>. "
    "Treat each placeholder as the real value it stands for and write it exactly as given (with the angle brackets) "
    "wherever you would use that value. Do not invent new placeholders."
)


def add_placeholder_hint(body: JsonDict, results: list[RedactionResult]) -> JsonDict:
    """Prepend the hint as a system message when at least one placeholder was sent upstream."""
    if not any(e.action.value == "pseudonymize" for r in results for e in r.entities):
        return body
    return {**body, "messages": [{"role": "system", "content": PLACEHOLDER_HINT}, *body.get("messages", [])]}


def merge_results(results: list[RedactionResult], session_id: str, policy: str) -> RedactionResult:
    """One summary for the whole request (audit log, response headers, block decision)."""
    merged = RedactionResult(text="", session_id=session_id, policy=policy)
    for result in results:
        merged.entities.extend(result.entities)
        for reason in result.block_reasons:
            if reason not in merged.block_reasons:
                merged.block_reasons.append(reason)
        merged.detector_errors.update(result.detector_errors)
        for key, value in result.timings_ms.items():
            merged.timings_ms[key] = round(merged.timings_ms.get(key, 0.0) + value, 2)
    merged.blocked = bool(merged.block_reasons)
    return merged


def _restore_arguments(session: ShieldSession, arguments: str, report: RestoreReport) -> str:
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        text, partial = session.restore_with_report(arguments, json_string=True)
        _add(report, partial)
        return text

    def restore(value: str) -> str:
        text, partial = session.restore_with_report(value)
        _add(report, partial)
        return text

    return json.dumps(_map_strings(parsed, restore), ensure_ascii=False)


def _add(total: RestoreReport, part: RestoreReport) -> None:
    total.restored += part.restored
    total.synthetic_restored += part.synthetic_restored
    total.unknown.extend(part.unknown)


def restore_chat_response(session: ShieldSession, response: JsonDict) -> tuple[JsonDict, RestoreReport]:
    restored = copy.deepcopy(response)
    report = RestoreReport()
    for choice in restored.get("choices", []):
        message = choice.get("message") or {}
        if isinstance(message.get("content"), str):
            text, partial = session.restore_with_report(message["content"])
            message["content"] = text
            _add(report, partial)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if isinstance(function.get("arguments"), str):
                function["arguments"] = _restore_arguments(session, function["arguments"], report)
    return restored, report


@dataclass
class StreamTransformer:
    """Restores streamed `chat.completion.chunk`s; placeholders split across chunks are held back briefly."""

    session: ShieldSession
    _content: dict[int, StreamRestorer] = field(default_factory=dict)
    _arguments: dict[tuple[int, int], StreamRestorer] = field(default_factory=dict)
    report: RestoreReport = field(default_factory=RestoreReport)
    raw_text: list[str] = field(default_factory=list)
    """The upstream text as received (with placeholders), for the playground and debugging."""

    def _content_restorer(self, choice: int) -> StreamRestorer:
        if choice not in self._content:
            self._content[choice] = self.session.stream_restorer()
        return self._content[choice]

    def _argument_restorer(self, choice: int, call: int) -> StreamRestorer:
        if (choice, call) not in self._arguments:
            self._arguments[(choice, call)] = self.session.stream_restorer(json_string=True)
        return self._arguments[(choice, call)]

    def transform(self, chunk: JsonDict) -> JsonDict:
        chunk = copy.deepcopy(chunk)
        for choice in chunk.get("choices", []):
            index = int(choice.get("index", 0))
            delta = choice.setdefault("delta", {})
            if isinstance(delta.get("content"), str) and delta["content"]:
                self.raw_text.append(delta["content"])
                delta["content"] = self._content_restorer(index).push(delta["content"])
            for call in delta.get("tool_calls") or []:
                function = call.get("function") or {}
                if isinstance(function.get("arguments"), str) and function["arguments"]:
                    restorer = self._argument_restorer(index, int(call.get("index", 0)))
                    function["arguments"] = restorer.push(function["arguments"])
            if choice.get("finish_reason"):
                self._flush_into(index, delta)
        return chunk

    def _flush_into(self, choice: int, delta: JsonDict) -> None:
        if choice in self._content:
            tail = self._content[choice].flush()
            if tail:
                delta["content"] = (delta.get("content") or "") + tail
        for (owner, call_index), restorer in self._arguments.items():
            if owner != choice:
                continue
            tail = restorer.flush()
            if tail:
                delta.setdefault("tool_calls", []).append({"index": call_index, "function": {"arguments": tail}})

    def finish(self) -> JsonDict | None:
        """Flush anything still held (upstream ended without a finish_reason): one extra chunk or None."""
        pending: dict[int, JsonDict] = {}
        for choice in {*self._content, *(owner for owner, _ in self._arguments)}:
            delta: JsonDict = {}
            self._flush_into(choice, delta)
            if delta:
                pending[choice] = delta
        self._collect_reports()
        if not pending:
            return None
        return {
            "object": "chat.completion.chunk",
            "choices": [{"index": i, "delta": d, "finish_reason": None} for i, d in sorted(pending.items())],
        }

    def _collect_reports(self) -> None:
        for restorer in [*self._content.values(), *self._arguments.values()]:
            _add(self.report, restorer.report)
            restorer.report = RestoreReport()
