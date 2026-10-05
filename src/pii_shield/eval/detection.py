"""Detection quality on the gold set: P/R/F1 per entity type, per detector configuration, plus latency."""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from pii_shield.detect.pipeline import DetectionPipeline
from pii_shield.entities import Span
from pii_shield.eval.config import DetectorConfig
from pii_shield.eval.gold import GoldDoc, gold_stats
from pii_shield.eval.metrics import PRF, MatchMode, SpanRef, aggregate, match, percentile, score_document
from pii_shield.policy import DetectorToggles, Policy


class ModeScores(BaseModel):
    overall: PRF
    by_type: dict[str, PRF]


class SplitScores(BaseModel):
    documents: int
    strict: PRF
    partial: ModeScores
    untyped: PRF


class ErrorExample(BaseModel):
    doc: str
    kind: str
    """"missed" (false negative) or "spurious" (false positive), partial matching."""
    type: str
    value: str


class ConfigResult(BaseModel):
    name: str
    label: str
    layers: list[str]
    models: dict[str, str] = Field(default_factory=dict)
    strict: ModeScores
    partial: ModeScores
    untyped: PRF
    """Any-overlap, any-type: share of gold PII that would be redacted at all."""
    holdout: SplitScores | None = None
    """The same metrics on the held-out split only (documents written after the detectors were frozen)."""
    negatives_with_findings: int
    negatives: int
    latency_ms: dict[str, float]
    layer_latency_ms: dict[str, float]
    detector_errors: int
    errors: list[ErrorExample] = Field(default_factory=list)


class SubsetReport(BaseModel):
    """All configurations, including the LLM layer, on the same subset of the gold set."""

    documents: int
    every: int
    entities: int
    configs: list[ConfigResult]


class DetectionReport(BaseModel):
    created: str
    policy: str
    gold: dict[str, object]
    configs: list[ConfigResult]
    llm_subset: SubsetReport | None = None
    ner_models: list[dict[str, object]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _refs(spans: Sequence[Span]) -> list[SpanRef]:
    return [SpanRef(s.type.value, s.start, s.end) for s in spans]


def evaluate_config(
    docs: list[GoldDoc],
    pipeline: DetectionPipeline,
    policy: Policy,
    config: DetectorConfig,
    *,
    models: dict[str, str] | None = None,
    workers: int = 1,
) -> ConfigResult:
    toggled = policy.model_copy(
        update={"detectors": DetectorToggles(patterns=config.patterns, ner=config.ner, llm=config.llm)}
    )

    def run(doc: GoldDoc) -> tuple[list[Span], float, dict[str, float], int]:
        started = time.perf_counter()
        outcome = pipeline.detect(doc.text, toggled)
        return outcome.spans, (time.perf_counter() - started) * 1000, outcome.timings_ms, len(outcome.errors)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(run, docs))
    else:
        runs = [run(doc) for doc in docs]

    per_doc: dict[MatchMode, list[dict[str, PRF]]] = {mode: [] for mode in MatchMode}
    errors: list[ErrorExample] = []
    layer_totals: dict[str, list[float]] = {}
    latencies: list[float] = []
    detector_errors = 0
    negatives_with_findings = 0
    for doc, (spans, latency, timings, error_count) in zip(docs, runs, strict=True):
        gold = [SpanRef(e.type.value, e.start, e.end) for e in doc.entities]
        predicted = _refs(spans)
        for mode in MatchMode:
            per_doc[mode].append(score_document(gold, predicted, mode))
        latencies.append(latency)
        for layer, ms in timings.items():
            layer_totals.setdefault(layer, []).append(ms)
        detector_errors += error_count
        if not doc.entities and spans:
            negatives_with_findings += 1
        errors.extend(_error_examples(doc, spans))

    holdout_idx = [i for i, doc in enumerate(docs) if "holdout" in doc.tags]
    holdout = None
    if holdout_idx:
        h_partial, h_partial_types = aggregate(per_doc[MatchMode.PARTIAL][i] for i in holdout_idx)
        holdout = SplitScores(
            documents=len(holdout_idx),
            strict=aggregate(per_doc[MatchMode.STRICT][i] for i in holdout_idx)[0],
            partial=ModeScores(overall=h_partial, by_type=h_partial_types),
            untyped=aggregate(per_doc[MatchMode.UNTYPED][i] for i in holdout_idx)[0],
        )
    strict_overall, strict_types = aggregate(per_doc[MatchMode.STRICT])
    partial_overall, partial_types = aggregate(per_doc[MatchMode.PARTIAL])
    untyped_overall, _ = aggregate(per_doc[MatchMode.UNTYPED])
    return ConfigResult(
        name=config.name,
        label=config.label,
        layers=[layer for layer, on in (("patterns", config.patterns), ("ner", config.ner), ("llm", config.llm)) if on],
        models=models or {},
        strict=ModeScores(overall=strict_overall, by_type=strict_types),
        partial=ModeScores(overall=partial_overall, by_type=partial_types),
        untyped=untyped_overall,
        holdout=holdout,
        negatives_with_findings=negatives_with_findings,
        negatives=sum(1 for doc in docs if not doc.entities),
        latency_ms={
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.5), 2),
            "p95": round(percentile(latencies, 0.95), 2),
        },
        layer_latency_ms={layer: round(sum(v) / len(v), 2) for layer, v in layer_totals.items()},
        detector_errors=detector_errors,
        errors=errors,
    )


def _error_examples(doc: GoldDoc, spans: Sequence[Span]) -> list[ErrorExample]:
    gold = [SpanRef(e.type.value, e.start, e.end) for e in doc.entities]
    predicted = _refs(spans)
    pairs = match(gold, predicted, MatchMode.PARTIAL)
    matched_gold = {gi for gi, _ in pairs}
    matched_pred = {pi for _, pi in pairs}
    out = [
        ErrorExample(doc=doc.id, kind="missed", type=e.type.value, value=e.value)
        for gi, e in enumerate(doc.entities)
        if gi not in matched_gold
    ]
    out += [
        ErrorExample(doc=doc.id, kind="spurious", type=s.type.value, value=s.text)
        for pi, s in enumerate(spans)
        if pi not in matched_pred
    ]
    return out


def new_report(docs: list[GoldDoc], policy: str) -> DetectionReport:
    return DetectionReport(
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        policy=policy,
        gold=gold_stats(docs),
        configs=[],
    )
