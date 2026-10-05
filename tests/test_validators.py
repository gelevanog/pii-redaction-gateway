import pytest

from pii_shield.detect import validators as v


@pytest.mark.parametrize(
    "number", ["4111 1111 1111 1111", "4526018159083012", "3486 252760 18954", "5537-8657-9754-3235"]
)
def test_luhn_valid_cards(number: str) -> None:
    assert v.luhn_valid(number)
    assert v.is_credit_card(number)


@pytest.mark.parametrize(
    "number", ["4111 1111 1111 1112", "4685 9952 8907 8667", "1234 5678 9012 3456", "0000 0000 0000 0000"]
)
def test_invalid_cards(number: str) -> None:
    assert not v.is_credit_card(number)


def test_card_needs_known_issuer_and_length() -> None:
    assert not v.is_credit_card("9" * 15 + "5")  # no issuer prefix 9
    assert not v.is_credit_card("4111 1111 111")  # too short


def test_luhn_check_digit_makes_valid_number() -> None:
    assert v.luhn_valid("411111111111111" + v.luhn_check_digit("411111111111111"))


@pytest.mark.parametrize(
    "iban",
    ["DE89 3704 0044 0532 0130 00", "GB29NWBK60161331926819", "es9121000418450200051332", "NL91 ABNA 0417 1643 00"],
)
def test_iban_valid(iban: str) -> None:
    assert v.iban_valid(iban)


@pytest.mark.parametrize(
    "iban", ["DE89 3704 0044 0532 0130 01", "DE12 3456 7890 1234 5678 90", "XX89370400440532013000", "DE893704"]
)
def test_iban_invalid(iban: str) -> None:
    assert not v.iban_valid(iban)


def test_iban_check_digits_roundtrip() -> None:
    assert v.iban_valid("DE" + v.iban_check_digits("DE", "370400440532013000") + "370400440532013000")


@pytest.mark.parametrize(
    "ssn,ok",
    [
        ("521-48-3907", True),
        ("000-12-3456", False),
        ("666-12-3456", False),
        ("912-12-3456", False),
        ("521-00-3907", False),
        ("521-48-0000", False),
        ("52148390", False),
    ],
)
def test_us_ssn_structure(ssn: str, ok: bool) -> None:
    assert v.us_ssn_valid(ssn) is ok


@pytest.mark.parametrize(
    "value,ok",
    [
        ("48291037F", True),
        ("48291037Q", False),
        ("X4718293G", True),
        ("Y4718293H", True),
        ("X4718293A", False),
        ("1234567Z", False),
    ],
)
def test_spanish_dni_nie(value: str, ok: bool) -> None:
    assert v.spanish_dni_valid(value) is ok


@pytest.mark.parametrize(
    "value,ok", [("121547280", True), ("3537.99.075", True), ("121547281", False), ("000000000", False)]
)
def test_dutch_bsn(value: str, ok: bool) -> None:
    assert v.dutch_bsn_valid(value) is ok


def test_ip_helpers() -> None:
    assert v.ip_address_valid("192.168.1.37") and v.ip_address_valid("2001:db8::1")
    assert not v.ip_address_valid("999.1.1.1")
    assert v.ip_is_loopback_or_unspecified("127.0.0.1") and v.ip_is_loopback_or_unspecified("::1")
    assert not v.ip_is_loopback_or_unspecified("10.0.0.23")


def test_secret_heuristics() -> None:
    assert v.looks_like_secret("Zx8vQ2mL9pR4tK7wB1nC", with_context=True)
    assert v.looks_like_secret("aB3dE6gH9jK2mN5pQ8rS1tV4wX7z", with_context=False)
    assert not v.looks_like_secret("3f2b8c1e-7a4d-4e9b-b1c2-5d6e7f8a9b0c", with_context=False)  # UUID
    assert not v.looks_like_secret("9f86d081884c7d659a2feaa0c55ad015a3bf4f1b", with_context=False)  # hex digest
    assert not v.looks_like_secret("internationalization", with_context=True)
    assert v.shannon_entropy("aaaa") == 0.0


def test_contains_email() -> None:
    assert v.contains_email("https://x.com/?e=anna%40gmail.com")
    assert not v.contains_email("https://brightloop.com/help")
