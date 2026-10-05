"""Span-level precision / recall / F1.

- strict: same type and exact character boundaries;
- partial: same type and any overlap (one-to-one matching, largest overlap first);
- untyped: any overlap, type ignored. This is the privacy view: a name redacted as ORGANIZATION is
  still not sent to the LLM.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import NamedTuple

from pydantic import BaseModel, computed_field


class MatchMode(StrEnum):
    STRICT = "strict"
    PARTIAL = "partial"
    UNTYPED = "untyped"


class SpanRef(NamedTuple):
    type: str
    start: int
    end: int


class PRF(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0

    def add(self, other: PRF) -> PRF:
        return PRF(tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn)


def _overlap(a: SpanRef, b: SpanRef) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def match(gold: Sequence[SpanRef], predicted: Sequence[SpanRef], mode: MatchMode) -> list[tuple[int, int]]:
    """Index pairs (gold, predicted) matched one-to-one under `mode`."""
    if mode is MatchMode.STRICT:
        remaining = {p: i for i, p in enumerate(predicted)}
        pairs = []
        for gi, g in enumerate(gold):
            pi = remaining.pop(g, None)
            if pi is not None:
                pairs.append((gi, pi))
        return pairs
    candidates = [
        (_overlap(g, p), gi, pi)
        for gi, g in enumerate(gold)
        for pi, p in enumerate(predicted)
        if _overlap(g, p) > 0 and (mode is MatchMode.UNTYPED or g.type == p.type)
    ]
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    pairs = []
    for _, gi, pi in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
        if gi not in used_gold and pi not in used_pred:
            used_gold.add(gi)
            used_pred.add(pi)
            pairs.append((gi, pi))
    return pairs


def score_document(gold: Sequence[SpanRef], predicted: Sequence[SpanRef], mode: MatchMode) -> dict[str, PRF]:
    """Per-type counts for one document (TP and FN under the gold type, FP under the predicted type)."""
    pairs = match(gold, predicted, mode)
    matched_gold = {gi for gi, _ in pairs}
    matched_pred = {pi for _, pi in pairs}
    counts: dict[str, PRF] = defaultdict(PRF)
    for gi, g in enumerate(gold):
        key = "ALL" if mode is MatchMode.UNTYPED else g.type
        counts[key] = counts[key].add(PRF(tp=1) if gi in matched_gold else PRF(fn=1))
    for pi, p in enumerate(predicted):
        if pi not in matched_pred:
            key = "ALL" if mode is MatchMode.UNTYPED else p.type
            counts[key] = counts[key].add(PRF(fp=1))
    return dict(counts)


def aggregate(per_document: Iterable[dict[str, PRF]]) -> tuple[PRF, dict[str, PRF]]:
    """Micro-averaged overall counts and per-type counts."""
    by_type: dict[str, PRF] = defaultdict(PRF)
    for counts in per_document:
        for kind, prf in counts.items():
            by_type[kind] = by_type[kind].add(prf)
    overall = PRF()
    for prf in by_type.values():
        overall = overall.add(prf)
    return overall, dict(sorted(by_type.items()))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
