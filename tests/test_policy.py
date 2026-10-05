from pathlib import Path

import pytest

from pii_shield.entities import EntityType
from pii_shield.policy import Action, AllowItem, PolicyError, PolicySet, load_policy
from pii_shield.shield import PACKAGED_POLICIES, Shield


def test_packaged_policies_load(policies: PolicySet) -> None:
    assert set(policies.names) == {"support-chat", "strict-finance", "analytics-irreversible", "natural-text"}
    strict = policies.get("strict-finance")
    assert strict.blocking_types == {EntityType.CREDIT_CARD, EntityType.US_SSN, EntityType.SECRET}
    assert strict.threshold(EntityType.PHONE) == 0.3 and strict.threshold(EntityType.EMAIL) == 0.4
    support = policies.get("support-chat")
    assert support.rule(EntityType.CREDIT_CARD).keep_last == 4
    assert support.is_allowed(EntityType.EMAIL, "Support@Brightloop.com")
    assert not support.is_allowed(EntityType.EMAIL, "anna@gmail.com")
    assert EntityType.ORGANIZATION not in policies.get("analytics-irreversible").protected_types


def test_invalid_policies_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nentities:\n  PERSON: {action: shred}\n")
    with pytest.raises(PolicyError):
        load_policy(bad)
    bad.write_text("name: bad\nunknown_key: 1\n")
    with pytest.raises(PolicyError):
        load_policy(bad)
    with pytest.raises(ValueError):
        AllowItem(value="x", pattern="y")
    with pytest.raises(PolicyError):
        PolicySet.from_dir(tmp_path / "empty", "x") if (tmp_path / "empty").mkdir() is None else None


def test_unknown_policy_name(policies: PolicySet) -> None:
    with pytest.raises(PolicyError):
        policies.get("nope")


def test_each_action_end_to_end(shield: Shield) -> None:
    text = "Anna Petrova, anna@gmail.com, card 4111 1111 1111 1111"
    assert shield.redact(text, policy="support-chat").text == "<PERSON_1>, <EMAIL_1>, card **** **** **** 1111"
    hashed = shield.redact(text, policy="analytics-irreversible").text
    assert hashed.startswith("<PERSON:") and "<EMAIL:" in hashed and "anna" not in hashed
    blocked = shield.redact(text, policy="strict-finance")
    assert blocked.blocked and blocked.block_reasons == ["CREDIT_CARD"] and "4111" not in blocked.text
    natural = shield.redact("Anna Petrova, anna@gmail.com", policy="natural-text")
    assert "<" not in natural.text and "Anna" not in natural.text and "@example." in natural.text


def test_policy_from_shield_create() -> None:
    shield = Shield.create("strict-finance")
    assert shield.policies.default_name == "strict-finance"
    assert shield.redact("SSN 521-48-3907").blocked
    assert Shield.create(PACKAGED_POLICIES / "support-chat.yaml").redact("mail a@b.co").text == "mail <EMAIL_1>"
    assert Action.BLOCK.value == "block"
