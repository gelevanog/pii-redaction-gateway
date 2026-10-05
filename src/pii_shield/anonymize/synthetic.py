"""Realistic synthetic stand-ins (Faker), format-preserving where it matters.

Values are deterministic per (session, entity) through a keyed seed, so "Anna Petrova" becomes the same
fake person in every message of a conversation. Wherever a reserved range exists the stand-in uses it,
so a synthetic value is never someone's real data: emails at example.com/.org/.net, IPs in the
documentation ranges (RFC 5737 / RFC 3849), SSNs with area 9xx (never issued). Phone numbers keep the
country code and layout with random digits; they are not guaranteed to be unallocated.
"""

from __future__ import annotations

import hashlib
import hmac
import random
import re
import string
from datetime import date, timedelta

import phonenumbers
from faker import Faker

from pii_shield.anonymize.normalize import name_tokens
from pii_shield.detect import validators as v
from pii_shield.entities import EntityType

_LOCALES = {"ru": "ru_RU", "de": "de_DE", "es": "es_ES", "en": "en_US"}
_fakers: dict[str, Faker] = {}


def _faker(locale: str) -> Faker:
    if locale not in _fakers:
        _fakers[locale] = Faker(locale)
    return _fakers[locale]


def guess_locale(value: str) -> str:
    if re.search(r"[А-Яа-яЁё]", value):
        return _LOCALES["ru"]
    if re.search(r"[äöüßÄÖÜ]|straße|strasse|\bweg\b|platz", value, re.IGNORECASE):
        return _LOCALES["de"]
    if re.search(r"[ñáéíóúÑÁÉÍÓÚ]|\bcalle\b|\bavenida\b|\bplaza\b", value, re.IGNORECASE):
        return _LOCALES["es"]
    return _LOCALES["en"]


def seed_for(secret: bytes, *parts: str) -> int:
    digest = hmac.new(secret, "\x1f".join(parts).encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big")


def _same_shape(value: str, rng: random.Random, keep_prefix: int = 0) -> str:
    """Replace digits with digits and letters with letters (same case), keep everything else."""
    out = []
    for index, char in enumerate(value):
        if index < keep_prefix:
            out.append(char)
        elif char.isdigit():
            out.append(rng.choice(string.digits))
        elif char.isascii() and char.isalpha():
            out.append(rng.choice(string.ascii_uppercase if char.isupper() else string.ascii_lowercase))
        else:
            out.append(char)
    return "".join(out)


def _person(value: str, fake: Faker) -> str:
    tokens = name_tokens(value)
    if len(tokens) <= 1:
        return str(fake.first_name())
    return f"{fake.first_name()} {fake.last_name()}"


def _phone(value: str, rng: random.Random) -> str:
    stripped = value.strip()
    if stripped.startswith(("+", "00")):
        for region in ("US", "GB", "DE"):
            try:
                number = phonenumbers.parse(stripped, region)
                keep = len(str(number.country_code)) + (1 if stripped.startswith("+") else 2)
                # keep "+49" and the first digit of the national number (mobile vs landline shape)
                return _same_shape(stripped, rng, keep_prefix=_index_after_digits(stripped, keep))
            except phonenumbers.NumberParseException:
                continue
    return _same_shape(stripped, rng, keep_prefix=_index_after_digits(stripped, 1))


def _index_after_digits(value: str, count: int) -> int:
    """Index just after the `count`-th digit-or-plus character (formatting characters don't count)."""
    seen = 0
    for index, char in enumerate(value):
        if char.isdigit() or char == "+":
            seen += 1
            if seen >= count:
                return index + 1
    return len(value)


def _card(value: str, rng: random.Random) -> str:
    digits = v.digits_only(value)
    body = digits[0] + "".join(rng.choice(string.digits) for _ in range(len(digits) - 2))
    new_digits = body + v.luhn_check_digit(body)
    iterator = iter(new_digits)
    return "".join(next(iterator) if char.isdigit() else char for char in value)


def _iban(value: str, rng: random.Random) -> str:
    compact = re.sub(r"[\s-]", "", value).upper()
    country, bban = compact[:2], compact[4:]
    new_bban = _same_shape(bban, rng).upper()
    new_compact = country + v.iban_check_digits(country, new_bban) + new_bban
    iterator = iter(new_compact)
    return "".join(next(iterator) if char.isalnum() else char for char in value)


def _ssn(value: str, rng: random.Random) -> str:
    new = f"9{rng.randint(0, 99):02d}{rng.randint(1, 99):02d}{rng.randint(1, 9999):04d}"
    iterator = iter(new)
    return "".join(next(iterator) if char.isdigit() else char for char in value)


def _national_id(value: str, rng: random.Random) -> str:
    compact = re.sub(r"[\s-]", "", value).upper()
    if re.fullmatch(r"[XYZ]?\d{7,8}[A-Z]", compact):
        prefix = compact[0] if compact[0] in "XYZ" else ""
        digits = "".join(rng.choice(string.digits) for _ in range(7 if prefix else 8))
        number = (str("XYZ".index(prefix)) + digits) if prefix else digits
        return prefix + digits + v.spanish_dni_letter(number)
    while True:  # Dutch BSN: random 9 digits passing the 11-test
        candidate = "".join(rng.choice(string.digits) for _ in range(9))
        if v.dutch_bsn_valid(candidate):
            iterator = iter(candidate)
            return "".join(next(iterator) if char.isdigit() else char for char in value)


def _date(value: str, rng: random.Random) -> str:
    """Shift the date by 1-3 years and some days, keeping the original format."""
    shift = timedelta(days=rng.randint(365, 3 * 365)) * rng.choice((-1, 1))
    numeric = re.fullmatch(r"(\d{1,2})([./-])(\d{1,2})\2(\d{2,4})", value)
    iso = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", value)
    if iso:
        try:
            shifted = date(int(iso[1]), int(iso[2]), int(iso[3])) + shift
            return f"{shifted.year:04d}-{shifted.month:02d}-{shifted.day:02d}"
        except ValueError:
            pass
    if numeric:
        a, sep, b, year = numeric.groups()
        try:
            day, month = (int(b), int(a)) if int(a) <= 12 < int(b) or sep == "/" else (int(a), int(b))
            full_year = int(year) + (1900 if len(year) == 2 else 0)
            shifted = date(full_year, month, day) + shift
        except ValueError:
            return _same_shape(value, rng)
        first, second = (
            (shifted.month, shifted.day) if (day, month) == (int(b), int(a)) else (shifted.day, shifted.month)
        )
        out_year = f"{shifted.year % 100:02d}" if len(year) == 2 else str(shifted.year)
        return f"{first:0{len(a)}d}{sep}{second:0{len(b)}d}{sep}{out_year}"
    # Textual month ("March 4, 1987", "4. März 1987"): new day and year, same words.
    day_replaced = re.sub(r"\b\d{1,2}\b", lambda _: str(rng.randint(1, 28)), value, count=1)
    return re.sub(r"\b(19|20)\d{2}\b", lambda m: str(int(m.group(0)) + rng.choice((-3, -2, -1, 1, 2, 3))), day_replaced)


def _secret(value: str, rng: random.Random) -> str:
    prefix = re.match(
        r"^(?:sk-(?:proj-|ant-[a-z0-9]+-|or-v1-)?|gh[pousr]_|github_pat_|glpat-|xox[abprs]-|AKIA|AIza|[spr]k_(?:live|test)_|eyJ)",
        value,
    )
    keep = prefix.end() if prefix else 0
    return _same_shape(value, rng, keep_prefix=keep)


def _ip(value: str, rng: random.Random) -> str:
    if ":" in value:
        return f"2001:db8::{rng.randint(1, 0xFFFF):x}"
    network = rng.choice(("192.0.2", "198.51.100", "203.0.113"))
    return f"{network}.{rng.randint(1, 254)}"


def synthesize(kind: EntityType, value: str, seed: int) -> str:
    """A realistic stand-in for `value`, deterministic for a given seed."""
    rng = random.Random(seed)
    locale = guess_locale(value)
    fake = _faker(locale)
    fake.seed_instance(seed)
    match kind:
        case EntityType.PERSON:
            return _person(value, fake)
        case EntityType.EMAIL:
            user = re.sub(r"[^a-z0-9.]", "", _faker("en_US").user_name().lower()) or "user"
            return f"{user}{rng.randint(1, 99)}@{rng.choice(('example.com', 'example.org', 'example.net'))}"
        case EntityType.PHONE:
            return _phone(value, rng)
        case EntityType.CREDIT_CARD:
            return _card(value, rng)
        case EntityType.IBAN:
            return _iban(value, rng)
        case EntityType.US_SSN:
            return _ssn(value, rng)
        case EntityType.NATIONAL_ID:
            return _national_id(value, rng)
        case EntityType.DATE_OF_BIRTH:
            return _date(value, rng)
        case EntityType.SECRET:
            return _secret(value, rng)
        case EntityType.IP_ADDRESS:
            return _ip(value, rng)
        case EntityType.URL:
            return f"https://example.com/u/{_faker('en_US').user_name().lower()}{rng.randint(10, 99)}"
        case EntityType.ADDRESS:
            if "," in value or "\n" in value:
                return str(fake.address()).replace("\n", ", ")
            return str(fake.street_address())
        case EntityType.ORGANIZATION:
            return str(fake.company())
        case EntityType.CUSTOM:
            return _same_shape(value, rng)
