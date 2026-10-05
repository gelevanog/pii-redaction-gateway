import re

from pii_shield.anonymize.restore import restore_text
from pii_shield.anonymize.strategies import Anonymizer, keyed_hash, mask
from pii_shield.anonymize.synthetic import synthesize
from pii_shield.detect import validators as v
from pii_shield.entities import EntityType
from pii_shield.policy import Action, EntityRule
from pii_shield.vault.session import SessionState

SECRET = b"s" * 32


def test_mask_keeps_separators_and_last_digits() -> None:
    assert mask("4111 1111 1111 1234", keep_last=4) == "**** **** **** 1234"
    assert mask("Summer2024!") == "**********!"


def test_hash_is_consistent_and_keyed() -> None:
    a = keyed_hash(EntityType.EMAIL, "Anna@Gmail.com", SECRET)
    assert a == keyed_hash(EntityType.EMAIL, "anna@gmail.com", SECRET)  # normalized before hashing
    assert a != keyed_hash(EntityType.EMAIL, "anna@gmail.com", b"other-key" * 4)
    assert re.fullmatch(r"<EMAIL:[0-9a-f]{8}>", a)


def test_pseudonymize_is_consistent_within_a_session() -> None:
    state = SessionState(session_id="s1")
    anonymizer = Anonymizer(SECRET)
    rule = EntityRule(action=Action.PSEUDONYMIZE)
    first = anonymizer.replace(state, EntityType.PHONE, "+49 30 12345678", rule)
    second = anonymizer.replace(state, EntityType.PHONE, "030 12345678", rule)  # same number, national format
    other = anonymizer.replace(state, EntityType.PHONE, "+44 7911 123456", rule)
    assert first == second == "<PHONE_1>" and other == "<PHONE_2>"


def test_keep_and_block_actions() -> None:
    state = SessionState(session_id="s1")
    anonymizer = Anonymizer(SECRET)
    assert anonymizer.replace(state, EntityType.ORGANIZATION, "Acme", EntityRule(action=Action.KEEP)) == "Acme"
    assert (
        anonymizer.replace(state, EntityType.CREDIT_CARD, "4111 1111 1111 1111", EntityRule(action=Action.BLOCK))
        == "[CREDIT_CARD BLOCKED]"
    )
    assert state.entries == []


def test_synthetic_values_are_format_preserving_and_safe() -> None:
    card = synthesize(EntityType.CREDIT_CARD, "4526 0181 5908 3012", seed=1)
    assert re.fullmatch(r"4\d{3} \d{4} \d{4} \d{4}", card) and v.luhn_valid(card) and card != "4526 0181 5908 3012"
    iban = synthesize(EntityType.IBAN, "DE89 3704 0044 0532 0130 00", seed=2)
    assert iban.startswith("DE") and v.iban_valid(iban) and len(iban) == len("DE89 3704 0044 0532 0130 00")
    ssn = synthesize(EntityType.US_SSN, "521-48-3907", seed=3)
    assert ssn.startswith("9") and not v.us_ssn_valid(ssn)  # area 9xx is never issued
    assert synthesize(EntityType.EMAIL, "anna@gmail.com", seed=4).split("@")[1] in {
        "example.com",
        "example.org",
        "example.net",
    }
    assert synthesize(EntityType.IP_ADDRESS, "81.2.69.142", seed=5).startswith(
        ("192.0.2.", "198.51.100.", "203.0.113.")
    )
    dni = synthesize(EntityType.NATIONAL_ID, "48291037F", seed=6)
    assert v.spanish_dni_valid(dni)
    assert synthesize(EntityType.DATE_OF_BIRTH, "1949-11-23", seed=7) != "1949-11-23"
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", synthesize(EntityType.DATE_OF_BIRTH, "12/03/1984", seed=8))


def test_synthetic_is_deterministic_per_seed() -> None:
    assert synthesize(EntityType.PERSON, "Anna Petrova", 42) == synthesize(EntityType.PERSON, "Anna Petrova", 42)
    assert synthesize(EntityType.PERSON, "Иван Смирнов", 1) != synthesize(EntityType.PERSON, "Anna Petrova", 1)


def test_synthetic_is_reversible_through_the_session() -> None:
    state = SessionState(session_id="s1")
    anonymizer = Anonymizer(SECRET)
    rule = EntityRule(action=Action.SYNTHETIC)
    fake_full = anonymizer.replace(state, EntityType.PERSON, "Anna Petrova", rule)
    fake_last = anonymizer.replace(state, EntityType.PERSON, "Petrova", rule)
    assert fake_last == fake_full.split()[-1]
    restored, report = restore_text(f"Dear {fake_full}, thanks. {fake_last} will get a call.", state)
    assert restored == "Dear Anna Petrova, thanks. Petrova will get a call."
    assert report.synthetic_restored == 2


def test_restore_tolerates_placeholder_variants_and_reports_unknown() -> None:
    state = SessionState(session_id="s1")
    entry = state.get_or_create(EntityType.EMAIL, "anna@gmail.com")
    state.assign_placeholder(entry)
    text = "Mail <EMAIL_1>, [EMAIL_1], {email_1}, &lt;EMAIL_1&gt;, < EMAIL_1 >, EMAIL_1, not email_1_x, ask <PERSON_7>."
    restored, report = restore_text(text, state)
    assert restored.count("anna@gmail.com") == 6
    assert "email_1_x" in restored
    assert report.unknown == ["<PERSON_7>"]


def test_restore_escapes_for_json_strings() -> None:
    state = SessionState(session_id="s1")
    entry = state.get_or_create(EntityType.PERSON, 'Anna "Ann" Petrova')
    state.assign_placeholder(entry)
    restored, _ = restore_text('{"name": "<PERSON_1>"}', state, json_string=True)
    assert restored == '{"name": "Anna \\"Ann\\" Petrova"}'
