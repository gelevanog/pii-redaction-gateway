"""Restoring placeholders that arrive split across stream chunks."""

import json

import pytest

from pii_shield.anonymize.restore import StreamRestorer
from pii_shield.entities import EntityType
from pii_shield.vault.session import SessionState

ANSWER = (
    "Hi <PERSON_1>, we'll email <EMAIL_1> and call PERSON_1's [PHONE_1] or &lt;PHONE_1&gt;. <PERSON_12> is unknown."
)
EXPECTED = (
    "Hi Anna Petrova, we'll email anna@gmail.com and call Anna Petrova's +44 7911 123456 or +44 7911 123456. "
    "<PERSON_12> is unknown."
)


@pytest.fixture
def state() -> SessionState:
    state = SessionState(session_id="stream")
    for kind, value in (
        (EntityType.PERSON, "Anna Petrova"),
        (EntityType.EMAIL, "anna@gmail.com"),
        (EntityType.PHONE, "+44 7911 123456"),
    ):
        state.assign_placeholder(state.get_or_create(kind, value))
    return state


def run(state: SessionState, chunks: list[str], *, json_string: bool = False) -> str:
    restorer = StreamRestorer(state, json_string=json_string)
    return "".join(restorer.push(chunk) for chunk in chunks) + restorer.flush()


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 13])
def test_fixed_size_chunks(state: SessionState, size: int) -> None:
    chunks = [ANSWER[i : i + size] for i in range(0, len(ANSWER), size)]
    assert run(state, chunks) == EXPECTED


def test_every_single_split_point(state: SessionState) -> None:
    for cut in range(len(ANSWER) + 1):
        assert run(state, [ANSWER[:cut], ANSWER[cut:]]) == EXPECTED, cut


def test_holdback_is_bounded(state: SessionState) -> None:
    restorer = StreamRestorer(state)
    emitted = restorer.push("Plain text that ends with an opening <PERS")
    assert emitted == "Plain text that ends with an opening "
    assert restorer.push("ON_1> done") == "Anna Petrova done"


def test_multi_digit_placeholder_is_not_cut_early(state: SessionState) -> None:
    surnames = ["Adams", "Baker", "Clark", "Davis", "Evans", "Fox", "Gray", "Hill", "Irwin", "Jones", "King"]
    for surname in surnames:  # <PERSON_2> ... <PERSON_12>
        state.assign_placeholder(state.get_or_create(EntityType.PERSON, f"Lee {surname}"))
    assert run(state, ["Hello <PERSON_1", "2>!"]) == "Hello Lee King!"
    assert run(state, ["Hello PERSON_1", "2!"]) == "Hello Lee King!"


def test_json_string_mode_escapes_values(state: SessionState) -> None:
    state.assign_placeholder(state.get_or_create(EntityType.ORGANIZATION, 'Acme "Logistics"'))
    arguments = '{"to": "<EMAIL_1>", "company": "<ORGANIZATION_1>"}'
    restored = run(state, [arguments[i : i + 4] for i in range(0, len(arguments), 4)], json_string=True)
    assert json.loads(restored) == {"to": "anna@gmail.com", "company": 'Acme "Logistics"'}


def test_synthetic_values_split_across_chunks() -> None:
    state = SessionState(session_id="syn")
    entry = state.get_or_create(EntityType.PERSON, "Anna Petrova")
    state.record_synthetic(entry, "Laura Jensen", "Anna Petrova")
    assert run(state, ["Dear Laura Jen", "sen, hi"]) == "Dear Anna Petrova, hi"
