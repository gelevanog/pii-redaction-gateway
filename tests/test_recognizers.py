"""Every pattern recognizer with positives and negatives. Fake secrets are assembled at runtime so the
repository never contains strings shaped like live credentials."""

import pytest

from pii_shield.detect.patterns import (
    CreditCardRecognizer,
    DateOfBirthRecognizer,
    EmailRecognizer,
    IbanRecognizer,
    IpAddressRecognizer,
    NationalIdRecognizer,
    PatternDetector,
    PhoneRecognizer,
    SecretRecognizer,
    UrlRecognizer,
    UsSsnRecognizer,
    is_personal_url,
)
from pii_shield.entities import EntityType


def texts(spans: list) -> list[str]:  # type: ignore[type-arg]
    return [s.text for s in spans]


def test_email_plain_and_obfuscated() -> None:
    found = texts(
        EmailRecognizer().find(
            "Mail anna.petrova@gmail.com or jsmith2009 at yahoo dot com or mike.d [at] pro [dot] net"
        )
    )
    assert found == ["anna.petrova@gmail.com", "jsmith2009 at yahoo dot com", "mike.d [at] pro [dot] net"]


def test_email_obfuscated_other_languages() -> None:
    assert texts(EmailRecognizer().find("an tobias punkt richter at gmx punkt de bitte")) == [
        "tobias punkt richter at gmx punkt de"
    ]


@pytest.mark.parametrize("text", ["look at the docs dot page for help", "we are at capacity", "meet at noon dot"])
def test_email_obfuscation_negatives(text: str) -> None:
    assert EmailRecognizer().find(text) == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("call me on +44 7911 123456 today", "+44 7911 123456"),
        ("reach me at (415) 555-2671 after six", "(415) 555-2671"),
        ("Telefon 0170 3829104", "0170 3829104"),
        ("mobile 07700 900461", "07700 900461"),  # Ofcom drama range: only "possible", accepted with a keyword
        ("tel +7 916 123-45-67", "+7 916 123-45-67"),
    ],
)
def test_phone_positives(text: str, expected: str) -> None:
    spans = [s for s in PhoneRecognizer().find(text) if s.score >= 0.5]
    assert texts(spans) == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "Order 4155550132 is late",  # order number context
        "Admitted 2026-03-02, discharged 2026-03-09",  # dates
        "tracking number is 4685 9952 8907 8667",  # fragment of a longer number
        "LAN address 192.168.1.37",  # an IP address
        "serial 2222 3333 4444",  # neither valid nor in phone context
    ],
)
def test_phone_negatives(text: str) -> None:
    assert [s for s in PhoneRecognizer().find(text) if s.score >= 0.5] == []


def test_credit_card() -> None:
    found = texts(
        CreditCardRecognizer().find("Card 4526 0181 5908 3012, bad 4111 1111 1111 1112, amex 3486 252760 18954")
    )
    assert found == ["4526 0181 5908 3012", "3486 252760 18954"]


def test_iban_stops_before_following_word() -> None:
    found = IbanRecognizer().find(
        "IBAN DE89 3704 0044 0532 0130 00 BIC COBADEFFXXX, invalid DE12 3456 7890 1234 5678 90"
    )
    assert texts(found) == ["DE89 3704 0044 0532 0130 00"]
    assert found[0].validated


def test_ip_addresses() -> None:
    text = "LAN 192.168.1.37, public 81.2.69.142, v6 2a02:8108:1a40:3e00:4d3:9e1f:61a2:7c0b, local 127.0.0.1"
    assert texts(IpAddressRecognizer().find(text)) == [
        "192.168.1.37",
        "81.2.69.142",
        "2a02:8108:1a40:3e00:4d3:9e1f:61a2:7c0b",
    ]


@pytest.mark.parametrize(
    "text", ["Firmware 2.14.0.3 broke it", "version 10.2.0.1", "mac 3C:71:BF:9A:12:4E", "at 10:42:15 UTC"]
)
def test_ip_negatives(text: str) -> None:
    assert IpAddressRecognizer().find(text) == []


@pytest.mark.parametrize(
    "url,personal",
    [
        ("https://www.linkedin.com/in/maria-gonzalez-pm", True),
        ("https://github.com/arjunmehta-dev", True),
        ("https://github.com/brightloop/sdk-python", False),
        ("https://crm.example.com/customers/88213", True),
        ("https://account.example.com/reset?token=abc&user=nora", True),
        ("https://brightloop.com/help/reset", False),
        ("https://user:pass@db.example.com/x", True),
        ("https://www.linkedin.com/", False),
    ],
)
def test_personal_urls(url: str, personal: bool) -> None:
    assert is_personal_url(url) is personal


def test_url_recognizer_strips_trailing_punctuation() -> None:
    assert texts(UrlRecognizer().find("Profile: https://calendly.com/oscar/30min.")) == [
        "https://calendly.com/oscar/30min"
    ]


def test_ssn() -> None:
    assert texts(UsSsnRecognizer().find("SSN 521-48-3907")) == ["521-48-3907"]
    assert texts(UsSsnRecognizer().find("my social security number is 392 41 8876")) == ["392 41 8876"]
    assert UsSsnRecognizer().find("call 392 41 8876") == []  # spaced, no context
    assert UsSsnRecognizer().find("SSN 666-12-3456") == []  # invalid area


def test_national_ids() -> None:
    text = "DNI 48291037F, NIE X4718293G, BSN: 121547280. Unrelated text far away from that keyword: 385280841"
    found = NationalIdRecognizer().find(text + " and pedido 48291037Q")
    assert [(s.text, s.source) for s in found] == [
        ("48291037F", "es_dni"),
        ("X4718293G", "es_dni"),
        ("121547280", "nl_bsn"),
    ]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("DOB: 12/03/1984", "12/03/1984"),
        ("born on March 4, 1987", "March 4, 1987"),
        ("geboren am 14.02.1988", "14.02.1988"),
        ("fecha de nacimiento 3 de mayo de 1995", "3 de mayo de 1995"),
        ("дата рождения 12.07.1985", "12.07.1985"),
        ("My birthday is October 9, 1979", "October 9, 1979"),
    ],
)
def test_date_of_birth(text: str, expected: str) -> None:
    assert texts(DateOfBirthRecognizer().find(text)) == [expected]


def test_date_without_birth_context_is_not_dob() -> None:
    assert DateOfBirthRecognizer().find("I bought it on 12/03/2024 and it broke on 05/09/2024") == []


def test_secrets_known_prefixes() -> None:
    github = "gh" + "p_" + "R2d9KfL0pQs8XnB4vZ1mT7wYc3HjE6aU5iO"
    openai_like = "sk-" + "proj-" + "Fq83mZpL2xVt7NcW1bRe9KdA"
    stripe_like = "sk_" + "live_" + "4kT9pL2xQm8vB7nR3cZ1"
    text = f"token {github} key={openai_like} aws AKIAIOSFODNN7EXAMPLE stripe {stripe_like}"
    found = texts(SecretRecognizer().find(text))
    for secret in (github, openai_like, "AKIAIOSFODNN7EXAMPLE", stripe_like):
        assert secret in found


def test_secrets_private_key_block() -> None:
    begin, end = "-----BEGIN " + "RSA PRIVATE KEY-----", "-----END " + "RSA PRIVATE KEY-----"
    text = f"{begin}\nTUlJQm9n_not_a_real_key\n{end}"
    spans = SecretRecognizer().find(text)
    assert text in texts(spans)  # the inner high-entropy line is also a candidate; overlap resolution keeps the block
    from pii_shield.detect.merge import resolve_overlaps

    assert texts(resolve_overlaps(text, spans)) == [text]


@pytest.mark.parametrize(
    "text,secret",
    [
        ("DATABASE_URL=postgres://app:Xk9#mT2vQp!@db:5432/x", "Xk9#mT2vQp!"),
        ("Authorization: Bearer eyJhbGciOi.eyJ1aWQiOjQ4.dGVzdC1zaWd", "eyJhbGciOi.eyJ1aWQiOjQ4.dGVzdC1zaWd"),
        ("password: Summer2024!", "Summer2024!"),
        ("I tried my password Tulip!Garden2023 twice", "Tulip!Garden2023"),
        ("Passwort für den Testzugang: Sommer2026!Haus", "Sommer2026!Haus"),
        ("retrying webhook with token Zx8vQ2mL9pR4tK7wB1nC after 401", "Zx8vQ2mL9pR4tK7wB1nC"),
    ],
)
def test_secret_context_patterns(text: str, secret: str) -> None:
    assert secret in texts(SecretRecognizer().find(text))


@pytest.mark.parametrize(
    "text",
    [
        "device_id=3f2b8c1e-7a4d-4e9b-b1c2-5d6e7f8a9b0c firmware=4.1.2",
        "commit 1f3a9c7e2b4d6f8a0c1e3b5d7f9a2c4e6b8d0f13 merged",
        "please reset my password reset link",
        "internationalization",
    ],
)
def test_secret_negatives(text: str) -> None:
    assert SecretRecognizer().find(text) == []


def test_pattern_detector_runs_every_recognizer() -> None:
    detector = PatternDetector.default()
    kinds = {s.type for s in detector.detect("a@b.co, 4111 1111 1111 1111, DE89 3704 0044 0532 0130 00, 10.0.0.1")}
    assert {EntityType.EMAIL, EntityType.CREDIT_CARD, EntityType.IBAN, EntityType.IP_ADDRESS} <= kinds
