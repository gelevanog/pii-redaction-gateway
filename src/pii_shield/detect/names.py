"""Name-variant propagation: once "Anna Petrova" is found, "Petrova" and "Anna" elsewhere are the same person.

NER models are good at full names and weaker at a lone surname later in the text ("Mr. Kovacs called
again"). Two sources of known names are propagated:

- PERSON spans detected in the same text;
- people already in the conversation's vault session (so message 5 is covered even if NER misses the
  surname there), plus exact earlier values of other types (an address mentioned again).

Tokens that are also everyday words ("Will", "Grace", "Mark", "May") are never propagated on their
own: "Will the update fix it?" must stay untouched even if a Will Turner appears later.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pii_shield.entities import EntityType, Span

AMBIGUOUS_NAME_WORDS = frozenset(
    {
        "will",
        "mark",
        "grace",
        "hope",
        "rose",
        "faith",
        "summer",
        "joy",
        "bill",
        "may",
        "june",
        "april",
        "august",
        "frank",
        "ivy",
        "dawn",
        "jack",
        "hunter",
        "pat",
        "sue",
        "art",
        "rob",
        "gene",
        "guy",
        "penny",
        "ruby",
        "amber",
        "holly",
        "iris",
        "lily",
        "daisy",
        "violet",
        "crystal",
        "sky",
        "storm",
        "river",
        "chase",
        "miles",
        "grant",
        "wade",
        "dean",
        "drew",
        "herb",
        "sunny",
        "honey",
        "cash",
        "king",
        "major",
        "page",
        "price",
        "rich",
        "young",
        "long",
        "brown",
        "white",
        "black",
        "green",
        "gray",
        "grey",
        "love",
        "star",
        "rosa",
        "victoria",
        "paris",
        "georgia",
        "jordan",
        "sydney",
        "austin",
        "florence",
        "carol",
        "christian",
        "earl",
        "ray",
        "rocky",
        "sandy",
    }
)
# Role nouns NER models like to tag as PERSON ("the customer", "IT manager", "support agent"). A PERSON span
# made only of these words is never a name; it would also garble system prompts ("Address the <PERSON_1>").
ROLE_WORDS = frozenset(
    {
        "customer",
        "customers",
        "client",
        "clients",
        "agent",
        "agents",
        "user",
        "users",
        "patient",
        "patients",
        "manager",
        "managers",
        "team",
        "staff",
        "support",
        "admin",
        "administrator",
        "operator",
        "assistant",
        "colleague",
        "colleagues",
        "caller",
        "sender",
        "recipient",
        "employee",
        "employees",
        "member",
        "representative",
        "rep",
        "owner",
        "tenant",
        "landlord",
        "buyer",
        "seller",
        "vendor",
        "contractor",
        "engineer",
        "technician",
        "installer",
        "director",
        "head",
        "ceo",
        "cfo",
        "cto",
        "coo",
        "it",
        "hr",
        "sales",
        "service",
        "desk",
        "officer",
        "lead",
        "chief",
        "executive",
        "person",
        "people",
        "sir",
        "madam",
        "dear",
        "kunde",
        "kundin",
        "mitarbeiter",
        "cliente",
        "agente",
        "usuario",
        "клиент",
        "оператор",
        "пользователь",
    }
)


def is_role_phrase(value: str) -> bool:
    tokens = [t.casefold() for t in re.findall(r"[^\W\d_]+", value)]
    return bool(tokens) and all(t in ROLE_WORDS for t in tokens)


_TOKEN = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)


def _covered(start: int, end: int, spans: Sequence[Span]) -> bool:
    return any(start < s.end and s.start < end for s in spans)


def name_variant_spans(
    text: str, names: Iterable[str], existing: Sequence[Span], score: float, source: str
) -> list[Span]:
    """Whole-word, capitalized occurrences of tokens of known names that no span covers yet."""
    found: list[Span] = []
    tokens: set[str] = set()
    for name in names:
        for token in _TOKEN.findall(name):
            if len(token) >= 3 and token[0].isupper() and token.casefold() not in AMBIGUOUS_NAME_WORDS:
                tokens.add(token)
    for token in sorted(tokens, key=len, reverse=True):
        for match in re.finditer(rf"(?<![\w-]){re.escape(token)}(?![\w-])", text):
            if _covered(match.start(), match.end(), [*existing, *found]):
                continue
            found.append(
                Span(
                    start=match.start(),
                    end=match.end(),
                    type=EntityType.PERSON,
                    text=match.group(0),
                    score=score,
                    source=source,
                )
            )
    return found


def known_value_spans(
    text: str, values: Iterable[tuple[EntityType, str]], existing: Sequence[Span], score: float = 0.9
) -> list[Span]:
    """Exact (case-insensitive, whole-word) re-occurrences of values already in the session vault."""
    found: list[Span] = []
    for kind, value in sorted(values, key=lambda item: len(item[1]), reverse=True):
        if len(value) < 3:
            continue
        for match in re.finditer(rf"(?<![\w-]){re.escape(value)}(?![\w-])", text, re.IGNORECASE):
            if _covered(match.start(), match.end(), [*existing, *found]):
                continue
            found.append(
                Span(start=match.start(), end=match.end(), type=kind, text=match.group(0), score=score, source="vault")
            )
    return found
