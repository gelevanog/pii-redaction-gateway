"""Property-based round trip: redact -> restore == original for the reversible strategy."""

import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pii_shield.detect import validators as v
from pii_shield.policy import REVERSIBLE_ACTIONS
from pii_shield.shield import RedactionResult, Shield

shield = Shield.create("support-chat")  # patterns only; no model needed

words = st.text(alphabet=string.ascii_lowercase, min_size=2, max_size=8).filter(lambda w: w not in {"at", "dot", "is"})
emails = st.builds(
    lambda user, domain: f"{user}@{domain}.com",
    st.from_regex(r"[a-z]{3,8}\.[a-z]{2,6}", fullmatch=True),
    st.from_regex(r"[a-z]{3,10}", fullmatch=True),
)
phones = st.builds(lambda a, b: f"+44 7911 {a:03d}{b:03d}", st.integers(100, 999), st.integers(0, 999))
ipv4 = st.builds(lambda a, b, c: f"81.{a}.{b}.{c}", st.integers(0, 255), st.integers(0, 255), st.integers(1, 254))


def _card(digits: str) -> str:
    body = "4" + digits
    return body + v.luhn_check_digit(body)


cards = st.builds(_card, st.from_regex(r"[0-9]{14}", fullmatch=True))
values = st.one_of(emails, phones, ipv4, cards.map(lambda c: f"card {c}"))


def expected_after_restore(text: str, result: RedactionResult) -> str:
    """Irreversible actions (mask) stay masked; everything reversible must come back exactly."""
    for entity in result.entities:
        if entity.action not in REVERSIBLE_ACTIONS:
            text = text.replace(entity.original, entity.replacement)
    return text


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.one_of(words, values), min_size=1, max_size=12))
def test_redact_restore_roundtrip(parts: list[str]) -> None:
    text = " ".join(parts) + "."
    for policy in ("support-chat", "natural-text"):  # placeholders, then realistic synthetic stand-ins
        with shield.session(policy=policy) as session:
            result = session.redact(text)
            assert session.restore(result.text) == expected_after_restore(text, result)
        for entity in result.entities:
            assert entity.original not in result.text
