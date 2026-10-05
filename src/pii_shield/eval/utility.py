"""Utility test with real models: does redaction hurt answer quality, and does restore put back the right values?

For each selected gold document the same task ("draft a reply") is run twice with the same free model:
directly on the original text (baseline), and through the gateway (redacted -> model -> restored).
A different free model judges the two answers blind (order randomized per document). Restore accuracy
counts placeholders in the raw model answers that resolved to a value of this conversation, and name
fidelity checks that the restored answer uses the customer's real name whenever the baseline did.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, Field

from pii_shield.anonymize.restore import PLACEHOLDER_PATTERN
from pii_shield.config import Settings
from pii_shield.dashboard.examples import TASKS
from pii_shield.detect.pipeline import DetectionPipeline
from pii_shield.entities import EntityType
from pii_shield.eval.config import UtilityConfig
from pii_shield.eval.gold import GoldDoc
from pii_shield.eval.leak import _digit_runs, _strings, is_leaked
from pii_shield.gateway.app import create_app
from pii_shield.gateway.runtime import build_runtime
from pii_shield.providers.base import ChatProvider, JsonDict, ProviderError
from pii_shield.providers.fake import RecordingProvider

JUDGE_SYSTEM = (
    "You are an impartial reviewer of customer-support replies. You compare two candidate replies to the same "
    "customer message and decide which is better, or whether they are of comparable quality."
)
JUDGE_SCHEMA: JsonDict = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["A", "B", "tie"]}, "reason": {"type": "string"}},
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def judge_prompt(message: str, reply_a: str, reply_b: str) -> str:
    return (
        f"Customer message:\n<<<\n{message}\n>>>\n\nReply A:\n<<<\n{reply_a}\n>>>\n\nReply B:\n<<<\n{reply_b}\n>>>\n\n"
        "Judge helpfulness, correctness with respect to the customer's message, and correct use of the customer's "
        "details (names, contact data, order facts). A reply that contains unresolved placeholders such as "
        "<PERSON_1> or wrong names is worse. Ignore differences in length or wording that do not change quality. "
        'Answer with JSON: {"verdict": "A" | "B" | "tie", "reason": "<one sentence>"}. Use "tie" when the replies '
        "are of comparable quality."
    )


class UtilityItem(BaseModel):
    doc: str
    verdict: str
    """From the redacted pipeline's point of view: better, equivalent or worse (or error)."""
    judge_reason: str = ""
    baseline: str = ""
    raw_redacted: str = ""
    restored: str = ""
    placeholders: int = 0
    restored_placeholders: int = 0
    unknown_placeholders: int = 0
    leftover_placeholders: int = 0
    baseline_uses_name: bool = False
    restored_uses_name: bool = False
    leaked_values: int = 0
    error: str | None = None


class NoiseFloor(BaseModel):
    """Same model, same prompt, two samples of the original-text answer, judged the same way.

    A judge rarely calls two answers of equal quality a tie, so this is the "worse" rate you get with no
    redaction at all; the redacted pipeline should be read against it, not against 100%.
    """

    documents: int
    verdicts: dict[str, int]
    equivalent_or_better: float | None


class UtilityReport(BaseModel):
    created: str
    task: str
    policy: str
    model: str
    judge_model: str
    served_models: dict[str, int] = Field(default_factory=dict)
    documents: int
    verdicts: dict[str, int]
    equivalent_or_better: float | None
    restore_accuracy: float | None
    placeholders_total: int
    unknown_placeholders: int
    leftover_placeholders: int
    name_fidelity: float | None
    name_fidelity_docs: int
    leaked_values: int
    items: list[UtilityItem]
    noise_floor: NoiseFloor | None = None


def select_documents(docs: list[GoldDoc], config: UtilityConfig) -> list[GoldDoc]:
    """Deterministic pick: documents with a named person, from the configured domains, spread over languages."""
    eligible = [
        d
        for d in docs
        if d.domain in config.domains and any(e.type is EntityType.PERSON for e in d.entities) and len(d.entities) >= 2
    ]
    eligible.sort(key=lambda d: hashlib.sha256(d.id.encode()).hexdigest())
    return sorted(eligible[: config.documents], key=lambda d: d.id)


def _content(response: JsonDict) -> str:
    return str(((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "")


def _uses_name(text: str, doc: GoldDoc) -> bool:
    for entity in doc.entities:
        if entity.type is EntityType.PERSON:
            for token in re.findall(r"[^\W\d_]{3,}", entity.value):
                if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text):
                    return True
    return False


async def run_utility(
    docs: list[GoldDoc],
    config: UtilityConfig,
    settings: Settings,
    pipeline: DetectionPipeline,
    model_provider: ChatProvider,
    judge_provider: ChatProvider,
    created: str,
) -> UtilityReport:
    selected = select_documents(docs, config)
    recorder = RecordingProvider(model_provider)
    runtime = build_runtime(settings, upstream=recorder, pipeline=pipeline)
    app = create_app(settings, runtime)
    instruction = TASKS[config.task]["prompt"]
    items: list[UtilityItem] = []
    served: dict[str, int] = {}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway", timeout=600
    ) as client:
        for doc in selected:
            body: dict[str, Any] = {
                "model": config.model.model,
                "messages": [{"role": "system", "content": instruction}, {"role": "user", "content": doc.text}],
                "max_tokens": config.max_tokens,
            }
            if config.model.provider == "openrouter":
                body["reasoning"] = {"effort": "low", "exclude": True}
            item = UtilityItem(doc=doc.id, verdict="error")
            try:
                baseline = await model_provider.complete(body)
                item.baseline = _content(baseline)
                served[str(baseline.get("model"))] = served.get(str(baseline.get("model")), 0) + 1
                before = len(recorder.requests)
                response = await client.post(f"/p/{config.policy}/v1/chat/completions", json=body)
                if response.status_code != 200:
                    raise ProviderError(f"gateway {response.status_code}: {response.text[:200]}")
                item.restored = _content(response.json())
                raw_response = recorder.responses[-1]
                served[str(raw_response.get("model"))] = served.get(str(raw_response.get("model")), 0) + 1
                item.raw_redacted = _content(raw_response)
                record = runtime.audit.recent(1)[0]
                item.placeholders = len(PLACEHOLDER_PATTERN.findall(item.raw_redacted))
                item.restored_placeholders = record.restored
                item.unknown_placeholders = record.unknown_placeholders
                item.leftover_placeholders = len(re.findall(r"<[A-Z_]+_\d+>", item.restored))
                sent = "\n".join(s for b in recorder.requests[before:] for s in _strings(b.get("messages", [])))
                digits = _digit_runs(sent)
                item.leaked_values = sum(1 for e in doc.entities if is_leaked(e.type, e.value, sent, digits))
                item.baseline_uses_name = _uses_name(item.baseline, doc)
                item.restored_uses_name = _uses_name(item.restored, doc)
                item.verdict, item.judge_reason = await _judge(judge_provider, config, doc, item)
            except ProviderError as exc:
                item.error = str(exc)[:300]
            items.append(item)

    noise = await _noise_floor(selected, items, config, model_provider, judge_provider) if config.noise_floor else None
    ok = [i for i in items if i.verdict in {"better", "equivalent", "worse"}]
    verdicts = {v: sum(1 for i in items if i.verdict == v) for v in ("better", "equivalent", "worse", "error")}
    placeholders = sum(i.placeholders for i in items if not i.error)
    unknown = sum(i.unknown_placeholders for i in items if not i.error)
    name_docs = [i for i in items if not i.error and i.baseline_uses_name]
    return UtilityReport(
        created=created,
        task=config.task,
        policy=config.policy,
        model=config.model.model,
        judge_model=config.judge.model,
        served_models=served,
        documents=len(selected),
        verdicts=verdicts,
        equivalent_or_better=(sum(1 for i in ok if i.verdict != "worse") / len(ok)) if ok else None,
        restore_accuracy=((placeholders - unknown) / placeholders) if placeholders else None,
        placeholders_total=placeholders,
        unknown_placeholders=unknown,
        leftover_placeholders=sum(i.leftover_placeholders for i in items if not i.error),
        name_fidelity=(sum(1 for i in name_docs if i.restored_uses_name) / len(name_docs)) if name_docs else None,
        name_fidelity_docs=len(name_docs),
        leaked_values=sum(i.leaked_values for i in items),
        items=items,
        noise_floor=noise,
    )


async def _noise_floor(
    docs: list[GoldDoc],
    items: list[UtilityItem],
    config: UtilityConfig,
    model_provider: ChatProvider,
    judge_provider: ChatProvider,
) -> NoiseFloor:
    instruction = TASKS[config.task]["prompt"]
    verdicts = dict.fromkeys(("better", "equivalent", "worse", "error"), 0)
    for doc, item in zip(docs, items, strict=True):
        if item.error:
            continue
        body: dict[str, Any] = {
            "model": config.model.model,
            "messages": [{"role": "system", "content": instruction}, {"role": "user", "content": doc.text}],
            "max_tokens": config.max_tokens,
            "seed": 2,  # a second, independent sample (and a different cache key)
        }
        if config.model.provider == "openrouter":
            body["reasoning"] = {"effort": "low", "exclude": True}
        try:
            second = _content(await model_provider.complete(body))
            pair = UtilityItem(doc=doc.id, verdict="", baseline=item.baseline, restored=second)
            verdict, _ = await _judge(judge_provider, config, doc, pair)
        except ProviderError:
            verdict = "error"
        verdicts[verdict] += 1
    judged = verdicts["better"] + verdicts["equivalent"] + verdicts["worse"]
    return NoiseFloor(
        documents=sum(verdicts.values()),
        verdicts=verdicts,
        equivalent_or_better=(verdicts["better"] + verdicts["equivalent"]) / judged if judged else None,
    )


async def _judge(judge: ChatProvider, config: UtilityConfig, doc: GoldDoc, item: UtilityItem) -> tuple[str, str]:
    redacted_first = int(hashlib.sha256(doc.id.encode()).hexdigest(), 16) % 2 == 0
    reply_a, reply_b = (item.restored, item.baseline) if redacted_first else (item.baseline, item.restored)
    body: dict[str, Any] = {
        "model": config.judge.model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": judge_prompt(doc.text, reply_a, reply_b)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "strict": True, "schema": JUDGE_SCHEMA},
        },
        "max_tokens": 4000,
    }
    if config.judge.provider == "openrouter":
        body["reasoning"] = {"effort": "low", "exclude": True}
        body["provider"] = {"require_parameters": True}
    response = await judge.complete(body)
    text = _content(response)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    verdict = str(data.get("verdict", "")).strip()
    reason = str(data.get("reason", ""))[:400]
    if verdict == "tie":
        return "equivalent", reason
    if verdict in {"A", "B"}:
        redacted_won = (verdict == "A") == redacted_first
        return ("better" if redacted_won else "worse"), reason
    return "error", f"unparseable judge answer: {text[:200]}"
