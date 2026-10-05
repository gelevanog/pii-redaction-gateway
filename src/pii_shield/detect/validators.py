"""Checksum and plausibility validators used by the pattern recognizers.

Every validator is a pure function on a string, so each one is tested with positives and negatives.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter

# ISO 13616 IBAN lengths per country (the countries a European support desk realistically sees).
IBAN_LENGTHS: dict[str, int] = {
    "AD": 24,
    "AE": 23,
    "AT": 20,
    "BE": 16,
    "BG": 22,
    "CH": 21,
    "CY": 28,
    "CZ": 24,
    "DE": 22,
    "DK": 18,
    "EE": 20,
    "ES": 24,
    "FI": 18,
    "FR": 27,
    "GB": 22,
    "GR": 27,
    "HR": 21,
    "HU": 28,
    "IE": 22,
    "IS": 26,
    "IT": 27,
    "LI": 21,
    "LT": 20,
    "LU": 20,
    "LV": 21,
    "MC": 27,
    "MT": 31,
    "NL": 18,
    "NO": 15,
    "PL": 28,
    "PT": 25,
    "RO": 24,
    "SA": 24,
    "SE": 24,
    "SI": 19,
    "SK": 24,
    "SM": 27,
    "TR": 26,
    "UA": 29,
}

_DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"


def digits_only(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def luhn_valid(number: str) -> bool:
    """Luhn (mod 10) checksum used by payment cards."""
    digits = digits_only(number)
    if not digits:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def luhn_check_digit(partial: str) -> str:
    """The digit that makes `partial + digit` Luhn-valid (used to build synthetic card numbers)."""
    for candidate in "0123456789":
        if luhn_valid(partial + candidate):
            return candidate
    raise AssertionError("unreachable: one of ten digits always satisfies Luhn")


def is_credit_card(value: str) -> bool:
    """13-19 digits, a known issuer prefix (Visa, Mastercard, Amex, Discover, JCB, UnionPay, Maestro) and Luhn."""
    digits = digits_only(value)
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    issuer = re.match(r"4|5[1-5]|2[2-7]|3[47]|6(?:011|5|4[4-9]|22)|35|62|50|5[6-9]|6[0-9]", digits)
    return issuer is not None and luhn_valid(digits)


def iban_valid(value: str) -> bool:
    """ISO 13616 IBAN: known country length and the ISO 7064 mod-97 check (remainder 1)."""
    iban = re.sub(r"[\s-]", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", iban):
        return False
    expected = IBAN_LENGTHS.get(iban[:2])
    if expected is None or len(iban) != expected:
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def iban_check_digits(country: str, bban: str) -> str:
    """Two check digits for a BBAN (used to build synthetic IBANs that pass validation)."""
    numeric = "".join(str(int(ch, 36)) for ch in (bban + country + "00").upper())
    return f"{98 - int(numeric) % 97:02d}"


def us_ssn_valid(value: str) -> bool:
    """Structural SSN rules: area not 000/666/9xx, group not 00, serial not 0000."""
    digits = digits_only(value)
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    # Well-known "example" numbers (123-45-6789, 078-05-1120) still pass: a customer who types one
    # into a ticket may well mean it, and over-redacting a fake SSN costs nothing.
    return not (area in {"000", "666"} or area.startswith("9") or group == "00" or serial == "0000")


def spanish_dni_valid(value: str) -> bool:
    """Spanish DNI (8 digits + letter) or NIE (X/Y/Z + 7 digits + letter): the letter is number mod 23."""
    raw = re.sub(r"[\s-]", "", value).upper()
    match = re.fullmatch(r"([XYZ]?)(\d{7,8})([A-Z])", raw)
    if not match:
        return False
    prefix, number, letter = match.groups()
    if prefix:
        if len(number) != 7:
            return False
        number = str("XYZ".index(prefix)) + number
    elif len(number) != 8:
        return False
    return _DNI_LETTERS[int(number) % 23] == letter


def spanish_dni_letter(number: str) -> str:
    return _DNI_LETTERS[int(number) % 23]


def dutch_bsn_valid(value: str) -> bool:
    """Dutch citizen service number (BSN): 9 digits passing the "elfproef" (11-test)."""
    digits = digits_only(value)
    if len(digits) != 9 or digits == "000000000":
        return False
    weights = [9, 8, 7, 6, 5, 4, 3, 2, -1]
    return sum(int(d) * w for d, w in zip(digits, weights, strict=True)) % 11 == 0


def ip_address_valid(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def ip_is_loopback_or_unspecified(value: str) -> bool:
    """127.0.0.1, ::1, 0.0.0.0 and :: describe the machine itself, not a person's device."""
    address = ipaddress.ip_address(value)
    return address.is_loopback or address.is_unspecified


def shannon_entropy(value: str) -> float:
    """Bits per character; random base64 tokens score ~4.5-6, English words ~2.5-3.5."""
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum(count / length * math.log2(count / length) for count in counts.values())


def looks_like_secret(token: str, *, with_context: bool) -> bool:
    """High-entropy token heuristic for API keys and passwords without a known prefix.

    Without a keyword nearby ("token", "api key", "password", ...) the bar is much higher, so git SHAs,
    UUIDs and long product codes are not flagged.
    """
    if len(token) < (12 if with_context else 24):
        return False
    classes = sum(bool(re.search(pattern, token)) for pattern in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]"))
    if re.fullmatch(r"[0-9a-fA-F-]+", token) and not with_context:
        return False  # hex digests and UUIDs are identifiers, not credentials, unless the text says otherwise
    if re.fullmatch(r"[A-Za-z]+", token) or re.fullmatch(r"\d+", token):
        return False
    entropy = shannon_entropy(token)
    if with_context:
        return classes >= 2 and entropy >= 3.0
    return classes >= 3 and entropy >= 4.0 and len(token) >= 24


_EMAIL_IN_TEXT = re.compile(r"[\w.+-]+(?:@|%40)[A-Za-z0-9-]+\.[A-Za-z]{2,}")


def contains_email(value: str) -> bool:
    """True if the string embeds an email address (also URL-encoded, e.g. in a query string)."""
    return bool(_EMAIL_IN_TEXT.search(value))
