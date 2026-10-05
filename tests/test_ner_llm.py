"""NER chunking (no model), the real GLiNER model (only when cached locally), and the LLM detector (scripted)."""

import json
from typing import Any

import pytest

from pii_shield.detect.llm import LlmDetector, locate, mask_known, parse_entities
from pii_shield.detect.ner import DEFAULT_NER_MODEL, GlinerDetector, chunk_text, model_cached
from pii_shield.detect.pipeline import DetectionPipeline
from pii_shield.entities import EntityType, Span
from pii_shield.policy import DetectorToggles, Policy


def test_chunk_text_keeps_offsets() -> None:
    text = " ".join(f"Sentence number {i} mentions Anna Petrova." for i in range(60))
    chunks = list(chunk_text(text, max_chars=200))
    assert len(chunks) > 5 and all(len(chunk) <= 200 for _, chunk in chunks)
    assert "".join(chunk for _, chunk in chunks) == text
    assert all(text[offset : offset + len(chunk)] == chunk for offset, chunk in chunks)
    assert list(chunk_text("short")) == [(0, "short")]


@pytest.mark.ner
@pytest.mark.skipif(not model_cached(DEFAULT_NER_MODEL), reason="GLiNER model not in the local Hugging Face cache")
def test_gliner_finds_names_addresses_and_orgs() -> None:
    detector = GlinerDetector()
    spans = detector.detect("Hi, this is Anna Petrova from Acme Logistics. Ship to 42 Baker Street, London NW1 6XE.")
    found = {(s.type, s.text) for s in spans}
    assert (EntityType.PERSON, "Anna Petrova") in found
    assert any(kind is EntityType.ADDRESS and "Baker Street" in text for kind, text in found)


class Scripted:
    def __init__(self, content: str) -> None:
        self.content = content
        self.bodies: list[dict[str, Any]] = []

    @property
    def label(self) -> str:
        return "scripted"

    @property
    def is_remote(self) -> bool:
        return False

    async def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(body)
        return {"model": "scripted", "choices": [{"message": {"role": "assistant", "content": self.content}}]}

    async def stream(self, body: dict[str, Any]):  # type: ignore[no-untyped-def]
        yield {}


def test_llm_detector_sees_masked_text_and_maps_back() -> None:
    text = "Anna (anna@gmail.com) said her colleague Bartholomew will call."
    email = Span(start=6, end=20, type=EntityType.EMAIL, text="anna@gmail.com", score=0.99)
    answer = json.dumps(
        {
            "entities": [
                {"type": "PERSON", "text": "Bartholomew"},
                {"type": "PERSON", "text": "Zelda"},
                {"type": "EMAIL", "text": "<EMAIL>"},
            ]
        }
    )
    provider = Scripted(answer)
    spans = LlmDetector(provider=provider, model="m:free").detect(text, [email])
    assert [(s.text, s.source) for s in spans] == [("Bartholomew", "llm")]  # "Zelda" does not occur: dropped
    sent = provider.bodies[0]["messages"][1]["content"]
    assert "anna@gmail.com" not in sent and "<EMAIL>" in sent
    assert provider.bodies[0]["response_format"]["type"] == "json_schema"


def test_llm_helpers() -> None:
    assert mask_known("a b c", [Span(start=2, end=3, type=EntityType.PERSON, text="b")]) == "a <PERSON> c"
    assert parse_entities('```json\n{"entities": [{"type": "PHONE", "text": "555 0101"}]}\n```') == [
        (EntityType.PHONE, "555 0101")
    ]
    assert parse_entities("not json") == []
    assert locate("Call Bo or bo", [(EntityType.PERSON, "Bo")], [], 0.7)[0].start == 5


def test_llm_layer_in_pipeline(support_policy: Policy) -> None:
    provider = Scripted(json.dumps({"entities": [{"type": "ADDRESS", "text": "the blue house behind the old mill"}]}))
    pipeline = DetectionPipeline(llm=LlmDetector(provider=provider, model="m:free"))
    policy = support_policy.model_copy(update={"detectors": DetectorToggles(patterns=True, ner=False, llm=True)})
    outcome = pipeline.detect("Deliver to the blue house behind the old mill, thanks", policy)
    assert [(s.type, s.source) for s in outcome.spans] == [(EntityType.ADDRESS, "llm")]
    assert "llm" in outcome.layers
