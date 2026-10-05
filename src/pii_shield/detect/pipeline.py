"""Layered detection: patterns -> NER -> deny-list -> thresholds and allow-list -> merge -> LLM (optional).

Every layer is timed separately; a failing layer is reported (not swallowed) so the shield can apply the
policy's fail-closed rule.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from pii_shield.anonymize.restore import PLACEHOLDER_PATTERN
from pii_shield.detect.llm import LlmDetector
from pii_shield.detect.merge import apply_thresholds, resolve_overlaps
from pii_shield.detect.names import known_value_spans, name_variant_spans
from pii_shield.detect.patterns import PatternDetector
from pii_shield.entities import EntityType, Span
from pii_shield.logging_config import get_logger
from pii_shield.policy import Policy

log = get_logger(__name__)


class Detector(Protocol):
    name: str

    def detect(self, text: str) -> list[Span]: ...


@dataclass
class DetectionOutcome:
    spans: list[Span]
    timings_ms: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    """Detector name -> error class and message (never the text)."""
    layers: list[str] = field(default_factory=list)
    """Detectors that actually ran."""


@dataclass
class DetectionPipeline:
    patterns: PatternDetector = field(default_factory=PatternDetector.default)
    ner: Detector | None = None
    llm: LlmDetector | None = None
    ner_expected: bool = False
    """True when NER was configured: a missing model is then a failure (fail-closed), not a deployment choice."""

    def available(self) -> dict[str, bool]:
        return {"patterns": True, "ner": self.ner is not None, "llm": self.llm is not None}

    def detect(
        self,
        text: str,
        policy: Policy,
        *,
        known_people: Sequence[str] = (),
        known_values: Sequence[tuple[EntityType, str]] = (),
    ) -> DetectionOutcome:
        """`known_people` / `known_values` come from the conversation's vault (see `names`)."""
        outcome = DetectionOutcome(spans=[])
        candidates: list[Span] = []

        if policy.detectors.patterns:
            candidates += self._run("patterns", self.patterns.detect, text, outcome)
        if policy.detectors.ner:
            if self.ner is None:
                if self.ner_expected:
                    outcome.errors["ner"] = "NerUnavailable: the NER model is not loaded"
            else:
                candidates += self._run("ner", self.ner.detect, text, outcome)
        candidates += self._deny_list(text, policy)

        spans = self._filter(text, candidates, policy)
        spans = self._propagate(text, spans, policy, known_people, known_values)

        if policy.detectors.llm:
            if self.llm is None:
                outcome.errors["llm"] = "LlmDetectorUnavailable: no LLM detector configured"
            else:
                llm = self.llm
                extra = self._run("llm", lambda t: llm.detect(t, spans), text, outcome)
                spans = self._filter(text, [*spans, *extra], policy)

        outcome.spans = spans
        return outcome

    def _run(self, name: str, detect: Callable[[str], list[Span]], text: str, outcome: DetectionOutcome) -> list[Span]:
        started = time.perf_counter()
        try:
            result = detect(text)
        except Exception as exc:  # a detector failure must be visible to the fail-closed rule
            outcome.errors[name] = f"{type(exc).__name__}: {str(exc)[:200]}"
            log.warning("detector.failed", detector=name, error=type(exc).__name__)
            result = []
        else:
            outcome.layers.append(name)
        outcome.timings_ms[name] = round((time.perf_counter() - started) * 1000, 2)
        return result

    def _propagate(
        self,
        text: str,
        spans: list[Span],
        policy: Policy,
        known_people: Sequence[str],
        known_values: Sequence[tuple[EntityType, str]],
    ) -> list[Span]:
        if not policy.link_name_variants:
            return spans
        people = [s.text for s in spans if s.type is EntityType.PERSON and s.source != "deny_list"]
        extra = name_variant_spans(text, [*people, *known_people], spans, score=0.85, source="name_variant")
        extra += known_value_spans(text, known_values, [*spans, *extra])
        if not extra:
            return spans
        return self._filter(text, [*spans, *extra], policy)

    @staticmethod
    def _deny_list(text: str, policy: Policy) -> list[Span]:
        spans = []
        for item in policy.deny_list:
            for match in item.regex().finditer(text):
                if match.end() > match.start():
                    spans.append(
                        Span(
                            start=match.start(),
                            end=match.end(),
                            type=item.entity,
                            text=match.group(0),
                            score=1.0,
                            source="deny_list",
                            validated=True,
                        )
                    )
        return spans

    @staticmethod
    def _filter(text: str, candidates: Sequence[Span], policy: Policy) -> list[Span]:
        # Placeholders already in the input (a system prompt explaining "<PERSON_1>", an earlier redacted
        # turn) are never PII themselves.
        placeholders = [(m.start(), m.end()) for m in PLACEHOLDER_PATTERN.finditer(text)]
        if placeholders:
            candidates = [
                s for s in candidates if not any(s.start < end and start < s.end for start, end in placeholders)
            ]
        kept = apply_thresholds(candidates, policy.thresholds(), policy.min_score)
        kept = [s for s in kept if s.source == "deny_list" or not policy.is_allowed(s.type, s.text)]
        resolved = resolve_overlaps(text, kept)
        # Trimming can change the surface ("Ms. Anna" -> "Anna"): re-check the allow-list on the final text.
        return [s for s in resolved if s.source == "deny_list" or not policy.is_allowed(s.type, s.text)]
