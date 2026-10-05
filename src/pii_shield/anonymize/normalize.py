"""Canonical keys for entity values, so the same thing gets the same placeholder however it is written.

"+49 30 1234567" and "030 1234567" are one phone, "ANNA.Petrova@Gmail.com" and
"anna dot petrova at gmail dot com" are one email, "DE89 3704 0044 ..." and "de8937040044..." one IBAN.
"""

from __future__ import annotations

import re
import unicodedata

import phonenumbers

from pii_shield.entities import EntityType

# Same obfuscation words as the email recognizer (detect/patterns.py).
_OBFUSCATED_AT = re.compile(
    r"\s*[\[({<]\s*(?:at|arroba)\s*[\])}>]\s*|\s+(?:at|arroba|собака)\s+|\s*\(@\)\s*|\s+@\s+", re.IGNORECASE
)
_OBFUSCATED_DOT = re.compile(
    r"\s*[\[({<]\s*(?:dot|punkt|punto|point)\s*[\])}>]\s*|\s+(?:dot|punkt|punto|point|точка)\s+", re.IGNORECASE
)
_NAME_TOKEN = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)


def _fold(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def normalize_email(value: str) -> str:
    value = _OBFUSCATED_AT.sub("@", value)
    value = _OBFUSCATED_DOT.sub(".", value)
    return re.sub(r"\s+", "", value).lower()


def normalize_phone(value: str) -> str:
    """National significant number: "+49 30 1234567", "0049 30 1234567" and "030 1234567" share one key.

    National formats are ambiguous across countries ("030..." is valid in Germany and the UK), so the key
    drops the country code and trunk prefix instead of guessing a region.
    """
    stripped = value.strip()
    if stripped.startswith(("+", "00")):
        try:
            number = phonenumbers.parse("+" + stripped.lstrip("+0") if stripped.startswith("00") else stripped, None)
            return str(number.national_number)
        except phonenumbers.NumberParseException:
            pass
    digits = "".join(ch for ch in stripped if ch.isdigit())
    if len(digits) == 11 and digits[0] in "18":
        digits = digits[1:]  # US "1 415 ...", Russian trunk "8 916 ..."
    return digits.lstrip("0")


def name_tokens(value: str) -> list[str]:
    """Casefolded word tokens of a name ("Anna-Lena O'Neil" -> ["anna-lena", "o'neil"])."""
    return [_fold(token).replace("’", "'") for token in _NAME_TOKEN.findall(value)]


def normalize(kind: EntityType, value: str) -> str:
    if kind is EntityType.EMAIL:
        return normalize_email(value)
    if kind is EntityType.PHONE:
        return normalize_phone(value)
    if kind in {EntityType.CREDIT_CARD, EntityType.IBAN, EntityType.US_SSN, EntityType.NATIONAL_ID}:
        return re.sub(r"[^0-9A-Za-z]", "", value).upper()
    if kind is EntityType.PERSON:
        return " ".join(name_tokens(value))
    if kind in {EntityType.ADDRESS, EntityType.ORGANIZATION, EntityType.CUSTOM}:
        return re.sub(r"[\s,.;:]+", " ", _fold(value)).strip()
    if kind in {EntityType.URL, EntityType.IP_ADDRESS}:
        return value.strip().rstrip("/").lower()
    return value.strip()
