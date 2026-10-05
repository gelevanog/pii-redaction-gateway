"""Optional LLM detector for contextual cases the patterns and NER miss. Off by default.

Privacy by construction: the model only sees the text *after* pattern and NER findings were replaced by
type tags (`<EMAIL>`, `<PERSON>`), and it returns exact substrings, never offsets (models are bad at
counting characters); the substrings are mapped back to the original text here, and anything that does
not occur verbatim is dropped as a hallucination. Point it at a model you trust with the residual text:
a self-hosted endpoint in production. The eval in this repo uses free OpenRouter models on the
synthetic gold set only.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pii_shield.entities import EntityType, Span
from pii_shield.providers.base import ChatProvider, JsonDict

_TYPES = [t.value for t in EntityType if t is not EntityType.CUSTOM]

SYSTEM_PROMPT = f"""You are a meticulous data-protection reviewer. Automated detectors already replaced the
personal data they found with tags such as <PERSON> or <EMAIL>. Find the personal data they MISSED.

Report only values that identify or contact a specific person (or are a credential):
- PERSON: names of people (first, last or full; no titles like Mr./Dr.)
- ADDRESS: street addresses (street + number, optionally postcode and city); not a city or country alone
- ORGANIZATION: names of companies, employers, schools, hospitals tied to the person
- EMAIL, PHONE, CREDIT_CARD, IBAN, IP_ADDRESS, US_SSN, NATIONAL_ID (passport, national ID, tax ID)
- URL: only links to a person's profile or carrying a personal identifier
- DATE_OF_BIRTH: only dates that are someone's birth date
- SECRET: passwords, API keys, tokens, private keys

Rules:
- Copy each value EXACTLY as it appears in the text (same spelling, case and punctuation).
- Never report the tags themselves (<PERSON>, <EMAIL>, ...) or text inside them.
- Do not report product names, generic job titles, public brands mentioned in passing, or ordinary words.
- If nothing was missed, return an empty list.
Allowed types: {", ".join(_TYPES)}."""

RESPONSE_SCHEMA: JsonDict = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"type": {"type": "string", "enum": _TYPES}, "text": {"type": "string"}},
                "required": ["type", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}


def mask_known(text: str, spans: Sequence[Span]) -> str:
    """The text with already-detected spans replaced by `<TYPE>` tags (what the LLM detector sees)."""
    out, cursor = [], 0
    for span in sorted(spans, key=lambda s: s.start):
        if span.start < cursor:
            continue
        out.append(text[cursor : span.start])
        out.append(f"<{span.type.value}>")
        cursor = span.end
    out.append(text[cursor:])
    return "".join(out)


def parse_entities(content: str) -> list[tuple[EntityType, str]]:
    """Parse the model's JSON answer; tolerate code fences and prose around the object."""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    found: list[tuple[EntityType, str]] = []
    for item in data.get("entities", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        kind, value = str(item.get("type", "")).upper(), str(item.get("text", "")).strip()
        if kind in _TYPES and len(value) >= 2 and not re.fullmatch(r"<[A-Z_]+>", value):
            found.append((EntityType(kind), value))
    return found


def locate(text: str, findings: list[tuple[EntityType, str]], existing: Sequence[Span], score: float) -> list[Span]:
    """Map reported substrings to every occurrence in `text` that no existing span covers."""
    spans: list[Span] = []
    for kind, value in findings:
        pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)")
        matches = list(pattern.finditer(text)) or list(re.finditer(re.escape(value), text, re.IGNORECASE))
        for match in matches:
            candidate = Span(
                start=match.start(), end=match.end(), type=kind, text=match.group(0), score=score, source="llm"
            )
            if not any(candidate.overlaps(s) for s in [*existing, *spans]):
                spans.append(candidate)
    return spans


@dataclass
class LlmDetector:
    provider: ChatProvider
    model: str
    score: float = 0.7
    """LLM findings carry no calibrated confidence; this fixed score lets policies threshold them."""
    max_tokens: int = 4000
    openrouter: bool = True
    """Add OpenRouter routing hints (low reasoning effort, only upstreams that honour response_format)."""
    name: str = "llm"

    def request_body(self, masked_text: str) -> JsonDict:
        body: JsonDict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Text:\n<<<\n{masked_text}\n>>>"},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "pii_entities", "strict": True, "schema": RESPONSE_SCHEMA},
            },
            "max_tokens": self.max_tokens,
        }
        if self.openrouter:
            body["reasoning"] = {"effort": "low", "exclude": True}
            body["provider"] = {"require_parameters": True}
        return body

    async def adetect(self, text: str, existing: Sequence[Span]) -> list[Span]:
        if not text.strip():
            return []
        response = await self.provider.complete(self.request_body(mask_known(text, existing)))
        content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return locate(text, parse_entities(content), existing, self.score)

    def detect(self, text: str, existing: Sequence[Span]) -> list[Span]:
        """Synchronous entry point (runs its own event loop; call `adetect` from async code)."""
        return asyncio.run(self.adetect(text, existing))
