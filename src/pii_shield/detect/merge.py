"""Span post-processing: trimming, overlap resolution and threshold filtering.

Several detectors look at the same text, so candidates overlap: an email that NER also calls a PERSON,
a phone-shaped run of digits inside a card number, "Anna" and "Anna Petrova" from two NER passes.
Resolution is deterministic:

1. Same-type overlaps are merged into one span covering both (the stronger evidence is kept).
2. Different-type overlaps: a validated span (checksum or library) beats an unvalidated one, then the
   higher score wins, then the longer span, then the earlier detector in priority order.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from pii_shield.detect.names import is_role_phrase
from pii_shield.entities import EntityType, Span

# Honorifics are not part of a name: "Ms. <PERSON_1>" keeps the text natural and the vault canonical.
_TITLES = re.compile(
    r"^(?:(?:mr|mrs|ms|miss|mx|dr|prof|sir|madam|herr|frau|fr|hr|sr|sra|srta|don|doña|dra|"
    r"господин|госпожа|г-н|г-жа)\.?\s+)+",
    re.IGNORECASE,
)
_TRAILING = re.compile(r"(?:['’]s|[\s.,;:!?)\]}\"'’-])+$")
_LEADING = re.compile(r"^[\s(\[{\"'“‘,.;:-]+")

_NER_LIKE = frozenset({EntityType.PERSON, EntityType.ADDRESS, EntityType.ORGANIZATION, EntityType.CUSTOM})

# Lower index = wins ties between equally scored, equally long spans of different types.
TYPE_PRIORITY: tuple[EntityType, ...] = (
    EntityType.SECRET,
    EntityType.CREDIT_CARD,
    EntityType.IBAN,
    EntityType.US_SSN,
    EntityType.NATIONAL_ID,
    EntityType.EMAIL,
    EntityType.URL,
    EntityType.IP_ADDRESS,
    EntityType.PHONE,
    EntityType.DATE_OF_BIRTH,
    EntityType.CUSTOM,
    EntityType.PERSON,
    EntityType.ADDRESS,
    EntityType.ORGANIZATION,
)
_PRIORITY = {kind: index for index, kind in enumerate(TYPE_PRIORITY)}


def trim_span(text: str, span: Span) -> Span | None:
    """Strip whitespace, punctuation, possessive 's and (for people) honorifics from the span edges."""
    start, end = span.start, span.end
    value = text[start:end]
    if span.type not in _NER_LIKE:
        # Pattern spans are exact by construction; only strip surrounding whitespace.
        start += len(value) - len(value.lstrip())
        end -= len(value) - len(value.rstrip())
        if (start, end) == (span.start, span.end):
            return span
        return span.model_copy(update={"start": start, "end": end, "text": text[start:end]}) if end > start else None
    leading = _LEADING.match(value)
    if leading:
        start += leading.end()
    value = text[start:end]
    if span.type in _NER_LIKE:
        trailing = _TRAILING.search(value)
        if trailing and span.type is not EntityType.ADDRESS:
            end -= len(trailing.group(0))
        elif span.type is EntityType.ADDRESS:
            end = start + len(value.rstrip(" ,;:"))
        if span.type is EntityType.PERSON:
            title = _TITLES.match(text[start:end])
            if title:
                start += title.end()
    if end - start < 2 or (span.type is EntityType.PERSON and end - start < 3):
        return None  # "Sí", "Да" as PERSON: one- and two-letter "names" are nearly always model noise
    if span.type is EntityType.PERSON and is_role_phrase(text[start:end]):
        return None  # "customer", "IT manager", "customer-support agent"
    if (start, end) == (span.start, span.end):
        return span
    return span.model_copy(update={"start": start, "end": end, "text": text[start:end]})


def _rank(span: Span) -> tuple[int, float, int, int]:
    # Sort key: better spans first.
    return (0 if span.validated else 1, -span.score, -span.length, _PRIORITY.get(span.type, 99))


def resolve_overlaps(text: str, spans: Iterable[Span]) -> list[Span]:
    """Return non-overlapping spans sorted by position."""
    trimmed = [t for s in spans if (t := trim_span(text, s)) is not None]
    merged = _merge_same_type(text, trimmed)
    chosen: list[Span] = []
    for candidate in sorted(merged, key=_rank):
        if not any(candidate.overlaps(kept) for kept in chosen):
            chosen.append(candidate)
    return sorted(chosen, key=lambda s: (s.start, s.end))


def _merge_same_type(text: str, spans: list[Span]) -> list[Span]:
    by_type: dict[EntityType, list[Span]] = {}
    for span in spans:
        by_type.setdefault(span.type, []).append(span)
    result: list[Span] = []
    for kind, group in by_type.items():
        group.sort(key=lambda s: (s.start, -s.end))
        current = group[0]
        for span in group[1:]:
            if span.start < current.end:
                best = min(current, span, key=_rank)
                start, end = min(current.start, span.start), max(current.end, span.end)
                current = best.model_copy(update={"start": start, "end": end, "text": text[start:end], "type": kind})
            else:
                result.append(current)
                current = span
        result.append(current)
    return result


def apply_thresholds(spans: Iterable[Span], thresholds: Mapping[EntityType, float], default: float) -> list[Span]:
    return [s for s in spans if s.score >= thresholds.get(s.type, default)]
