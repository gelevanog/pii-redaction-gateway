"""Pattern recognizers: a regex finds candidates, a validator (checksum, library or context) confirms them.

Each recognizer returns `Span`s with a score: validated values with supporting context score high,
plausible-but-unconfirmed ones score lower so a policy threshold can decide.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit

import phonenumbers

from pii_shield.detect import validators as v
from pii_shield.entities import EntityType, Span

DEFAULT_PHONE_REGIONS: tuple[str, ...] = ("US", "GB", "DE", "ES", "FR", "NL", "IT", "RU")


class Recognizer(Protocol):
    name: str

    def find(self, text: str) -> list[Span]: ...


def _before(text: str, start: int, width: int = 40) -> str:
    return text[max(0, start - width) : start].lower()


def _span(text: str, start: int, end: int, kind: EntityType, source: str, score: float, *, validated: bool) -> Span:
    return Span(start=start, end=end, type=kind, text=text[start:end], score=score, source=source, validated=validated)


# --------------------------------------------------------------------------------------------- email
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}(?![\w-])")
_TLDS = r"(?:com|org|net|io|co|de|es|fr|nl|it|ru|uk|eu|info|biz|me|dev|app|edu|gov|at|ch|be|pl|se|mx|ar|us)"
_AT = r"(?:\s*[\[({<]\s*(?:at|arroba)\s*[\])}>]\s*|\s+(?:at|arroba|собака)\s+|\s*\(@\)\s*|\s+@\s+)"
_DOT = r"(?:\s*[\[({<]\s*(?:dot|punkt|punto|point)\s*[\])}>]\s*|\s+(?:dot|punkt|punto|point|точка)\s+|\.)"
_OBFUSCATED_EMAIL = re.compile(
    rf"(?<![\w.])[A-Za-z0-9][\w+-]*(?:{_DOT}[\w+-]+)*?{_AT}[A-Za-z0-9-]+(?:{_DOT}[A-Za-z0-9-]+)*?{_DOT}{_TLDS}\b",
    re.IGNORECASE,
)


@dataclass
class EmailRecognizer:
    name: str = "email"

    def find(self, text: str) -> list[Span]:
        spans = [
            _span(text, m.start(), m.end(), EntityType.EMAIL, self.name, 0.99, validated=True)
            for m in _EMAIL.finditer(text)
        ]
        for m in _OBFUSCATED_EMAIL.finditer(text):
            value = m.group(0)
            # Needs an explicit obfuscation token ("at" / "[dot]"), otherwise it is a normal email found above.
            if "@" in value and not re.search(r"\b(?:dot|punkt|punto|point)\b|точка", value, re.IGNORECASE):
                continue
            spans.append(_span(text, m.start(), m.end(), EntityType.EMAIL, "email_obfuscated", 0.9, validated=False))
        return spans


# --------------------------------------------------------------------------------------------- phone
_PHONE_CONTEXT = re.compile(
    r"(phone|call|tel\b|tel\.|mobile|cell|whatsapp|signal|sms|text me|ring|fax|hotline|reach me|m\.|"
    r"telefon|handy|rufnummer|teléfono|telefono|móvil|movil|llámame|телефон|тел\.|моб|звоните|номер)"
)
_PHONE_NEGATIVE = re.compile(
    r"(order|invoice|tracking|ticket|case|ref\b|reference|serial|sku|account no|customer id|policy|"
    r"bestell|rechnung|pedido|factura|заказ|счёт|счет)[\s#:no.№-]*$"
)


_SSN_SHAPE = re.compile(r"\d{3}([- ])\d{2}\1\d{4}")
_INTERNATIONAL = re.compile(r"(?<![\w+])\+\d[\d ().-]{6,24}\d")
_DATE_SHAPE = re.compile(
    r"(?:19|20)\d{2}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./](?:19|20)\d{2}|(?:19|20)\d{2}-\d{3,5}"
)
_IPV4_SHAPE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")


def _is_fragment(text: str, start: int, end: int) -> bool:
    """True if digits continue right before or after the match ("4685 [9952 8907 8667]")."""
    before, after = text[max(0, start - 2) : start], text[end : end + 2]
    return bool(re.search(r"\d[ .-]?$", before) or re.match(r"^[ .-]?\d", after))


@dataclass
class PhoneRecognizer:
    """phonenumbers' matcher per region. Valid numbers score high; numbers that are only "possible"
    (right length, unallocated or fictional ranges such as UK 07700 900xxx) count when written in
    international format or next to a phone keyword: a redactor should not let them through."""

    regions: Sequence[str] = DEFAULT_PHONE_REGIONS
    name: str = "phone"

    def find(self, text: str) -> list[Span]:
        found: dict[tuple[int, int], Span] = {}
        scan = text
        for _ in range(3):
            # phonenumbers can swallow two numbers separated by a space into one invalid candidate and then
            # skip both; blanking out what was found and scanning again recovers the second one.
            before = len(found)
            self._scan(text, scan, found)
            if len(found) == before:
                break
            chars = list(text)
            for start, end in found:
                chars[start:end] = ["\n"] * (end - start)
            scan = "".join(chars)
        self._international(text, found)
        return list(found.values())

    def _international(self, text: str, found: dict[tuple[int, int], Span]) -> None:
        """ "+44 7911 236307 81.18.121.112": the matcher sees one impossible digit run. Shrink "+CC ..." candidates
        at separator boundaries until a possible number remains."""
        for m in _INTERNATIONAL.finditer(text):
            candidate = m.group(0)
            cuts = sorted({i for i, ch in enumerate(candidate) if ch in " .-"} | {len(candidate)}, reverse=True)
            for cut in cuts:
                piece = candidate[:cut].rstrip(" .-(")
                start, end = m.start(), m.start() + len(piece)
                if any(start < e and s < end for s, e in found):
                    break
                try:
                    number = phonenumbers.parse(piece, None)
                except phonenumbers.NumberParseException:
                    continue
                if phonenumbers.is_possible_number(number):  # longest first, so trailing groups are dropped
                    valid = phonenumbers.is_valid_number(number)
                    score = 0.95 if valid else 0.7
                    found[(start, end)] = _span(text, start, end, EntityType.PHONE, self.name, score, validated=valid)
                    break

    def _scan(self, text: str, scan: str, found: dict[tuple[int, int], Span]) -> None:
        for region in self.regions:
            for match in phonenumbers.PhoneNumberMatcher(scan, region, leniency=phonenumbers.Leniency.POSSIBLE):
                start, end = match.start, match.end
                if len(v.digits_only(match.raw_string)) < 7:
                    continue
                valid = phonenumbers.is_valid_number(match.number)
                previous = found.get((start, end))
                if previous is not None and (previous.validated or not valid):
                    continue
                raw = match.raw_string.strip()
                fragment = _is_fragment(text, start, end) and not raw.startswith(
                    ("+", "(")
                )  # "+"/"(" start a new number
                if _DATE_SHAPE.fullmatch(raw) or _IPV4_SHAPE.fullmatch(raw) or fragment:
                    continue  # a date, an IP address, or a piece of a longer number (card, IBAN, tracking id)
                before = _before(text, start, 30)
                international = raw.startswith(("+", "00"))
                context = bool(_PHONE_CONTEXT.search(before))
                if _PHONE_NEGATIVE.search(before) or _SSN_SHAPE.fullmatch(raw):
                    score = 0.3  # an order number, or 078-05-1120 which is shaped like an SSN
                elif valid:
                    score = 0.95 if context or international or match.raw_string.lstrip().startswith("(") else 0.8
                elif international or context:
                    score = 0.85 if international and context else 0.7
                else:
                    continue
                found[(start, end)] = _span(text, start, end, EntityType.PHONE, self.name, score, validated=valid)


# --------------------------------------------------------------------------------------- credit card
_CARD = re.compile(r"(?<![\d+-])(?:\d[ -]?){12,18}\d(?![\d-])")  # never right after "+": that is a phone


@dataclass
class CreditCardRecognizer:
    name: str = "credit_card"

    def find(self, text: str) -> list[Span]:
        spans = []
        for m in _CARD.finditer(text):
            # Greedy matching may swallow a following number ("4000 ... 0002 81"): try shorter cuts too.
            candidate = m.group(0)
            for cut in sorted({i for i, ch in enumerate(candidate) if ch in " -"} | {len(candidate)}, reverse=True):
                piece = candidate[:cut]
                if v.is_credit_card(piece):
                    end = m.start() + len(piece)
                    spans.append(_span(text, m.start(), end, EntityType.CREDIT_CARD, self.name, 0.97, validated=True))
                    break
        return spans


# ---------------------------------------------------------------------------------------------- IBAN
_IBAN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2}\d{2}(?:[ -]?[A-Za-z0-9]){10,30}")


@dataclass
class IbanRecognizer:
    name: str = "iban"

    def find(self, text: str) -> list[Span]:
        spans = []
        for m in _IBAN.finditer(text):
            # The regex is greedy and may swallow a following word ("... 3000 BIC"): try shorter cut points.
            candidate = m.group(0)
            for cut in range(len(candidate), 13, -1):
                if cut < len(candidate) and candidate[cut].isalnum() and candidate[cut - 1].isalnum():
                    continue  # only cut at a group boundary
                piece = candidate[:cut].rstrip(" -")
                if v.iban_valid(piece):
                    start = m.start()
                    spans.append(
                        _span(text, start, start + len(piece), EntityType.IBAN, self.name, 0.98, validated=True)
                    )
                    break
        return spans


# ------------------------------------------------------------------------------------------------ IP
_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w)(?!\.\d)")
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")


_VERSION_CONTEXT = re.compile(r"(firmware|version|release|upgrade|upgraded|rollback|update|\bsdk\b|\bbuild\b)")


@dataclass
class IpAddressRecognizer:
    name: str = "ip_address"

    def find(self, text: str) -> list[Span]:
        spans = []
        for m in _IPV4.finditer(text):
            before = _before(text, m.start(), 30)
            if re.search(r"(?:\bv|version|ver\.?|release|build)\s*$", before) or _VERSION_CONTEXT.search(before):
                continue
            if v.ip_address_valid(m.group(0)) and not v.ip_is_loopback_or_unspecified(m.group(0)):
                spans.append(_span(text, m.start(), m.end(), EntityType.IP_ADDRESS, self.name, 0.9, validated=True))
        for m in _IPV6.finditer(text):
            value = m.group(0)
            if (
                value.count(":") >= 2
                and re.search(r"[0-9A-Fa-f]", value)
                and v.ip_address_valid(value)
                and not v.ip_is_loopback_or_unspecified(value)
            ):
                spans.append(_span(text, m.start(), m.end(), EntityType.IP_ADDRESS, self.name, 0.9, validated=True))
        return spans


# ----------------------------------------------------------------------------------------------- URL
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"'`]+", re.IGNORECASE)
_SOCIAL_HOSTS = (
    "linkedin.com",
    "facebook.com",
    "fb.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "github.com",
    "gitlab.com",
    "t.me",
    "vk.com",
    "ok.ru",
    "xing.com",
    "tiktok.com",
    "youtube.com",
    "medium.com",
    "calendly.com",
    "wa.me",
    "threads.net",
    "bsky.app",
    "mastodon.social",
)
_CODE_HOSTS = frozenset({"github.com", "gitlab.com", "bitbucket.org"})
_NON_PROFILE_PATHS = re.compile(r"^/(?:about|help|legal|privacy|terms|login|signup|pricing|docs?|features)?/?$")
_PERSONAL_PATH = re.compile(
    r"/(?:users?|u|profiles?|people|members?|customers?|clients?|patients?|employees?|accounts?|in|pub|~)[/=]"
    r"|/~[\w.-]+|/@[\w.-]+",
    re.IGNORECASE,
)
_PERSONAL_QUERY = re.compile(
    r"[?&](?:e?mail|user(?:_?id|name)?|uid|login|token|session|sid|account|customer(?:_?id)?|phone|name|"
    r"first_?name|last_?name|patient(?:_?id)?|ssn|dob|auth|key|reset|invite)=",
    re.IGNORECASE,
)


def is_personal_url(url: str) -> bool:
    """A URL is personal data when it points at a person (profile, account page) or carries identifiers."""
    if "@" in url.split("?")[0].split("//", 1)[-1].split("/")[0]:
        return True  # credentials in the authority part: https://user:pass@host
    parts = urlsplit(url if "://" in url else "http://" + url)
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = parts.path or "/"
    if _PERSONAL_QUERY.search("?" + parts.query) or v.contains_email(url):
        return True
    if host in _CODE_HOSTS:
        # github.com/<user> is a profile; github.com/<owner>/<repo> is a project page
        return len([segment for segment in path.split("/") if segment]) == 1 and not _NON_PROFILE_PATHS.match(path)
    if any(host == social or host.endswith("." + social) for social in _SOCIAL_HOSTS):
        return not _NON_PROFILE_PATHS.match(path)
    return bool(_PERSONAL_PATH.search(path))


@dataclass
class UrlRecognizer:
    name: str = "url"

    def find(self, text: str) -> list[Span]:
        spans = []
        for m in _URL.finditer(text):
            value = m.group(0).rstrip(".,;:!?)]}")
            if is_personal_url(value):
                end = m.start() + len(value)
                spans.append(_span(text, m.start(), end, EntityType.URL, self.name, 0.9, validated=False))
        return spans


# ------------------------------------------------------------------------------------------- US SSN
_SSN = re.compile(r"(?<![\d-])(\d{3})([- ]?)(\d{2})\2(\d{4})(?![\d-])")
_SSN_CONTEXT = re.compile(r"(ssn|social security|soc\. sec|ss#|tax id|tin\b|itin)")


@dataclass
class UsSsnRecognizer:
    name: str = "us_ssn"

    def find(self, text: str) -> list[Span]:
        spans = []
        for m in _SSN.finditer(text):
            if not v.us_ssn_valid(m.group(0)):
                continue
            context = bool(_SSN_CONTEXT.search(_before(text, m.start(), 40)))
            separator = m.group(2)
            if separator == "-":
                score = 0.97 if context else 0.85
            elif context:
                score = 0.9
            else:
                continue  # 9 bare digits or space-separated groups without context: too ambiguous
            spans.append(_span(text, m.start(), m.end(), EntityType.US_SSN, self.name, score, validated=True))
        return spans


# ------------------------------------------------------------------------------------- national IDs
_DNI = re.compile(r"(?<![\w-])[XYZxyz]?[- ]?\d{7,8}[- ]?[A-Za-z](?![\w-])")
_BSN = re.compile(r"(?<![\d.])(?:\d{4}\.?\d{2}\.?\d{3}|\d{9})(?!\d)(?!\.\d)")
_BSN_CONTEXT = re.compile(r"(bsn|burgerservicenummer|sofi-?nummer|citizen service number|service number)")


@dataclass
class NationalIdRecognizer:
    """Non-US national IDs with checksums: Spanish DNI/NIE (mod-23 letter) and Dutch BSN (11-test, needs context)."""

    name: str = "national_id"

    def find(self, text: str) -> list[Span]:
        spans = []
        for m in _DNI.finditer(text):
            if v.spanish_dni_valid(m.group(0)):
                spans.append(_span(text, m.start(), m.end(), EntityType.NATIONAL_ID, "es_dni", 0.95, validated=True))
        for m in _BSN.finditer(text):
            if v.dutch_bsn_valid(m.group(0)) and _BSN_CONTEXT.search(_before(text, m.start(), 40)):
                spans.append(_span(text, m.start(), m.end(), EntityType.NATIONAL_ID, "nl_bsn", 0.95, validated=True))
        return spans


# ------------------------------------------------------------------------------------ date of birth
_MONTHS = (
    r"jan(?:uary|uar|\.)?|feb(?:ruary|ruar|\.)?|mar(?:ch|\.)?|märz|apr(?:il|\.)?|may|mai|jun(?:e|i|\.)?|"
    r"jul(?:y|i|\.)?|aug(?:ust|\.)?|sep(?:tember|t|\.)?|o[ck]t(?:ober|\.)?|nov(?:ember|\.)?|de[cz](?:ember|\.)?|"
    r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|"
    r"января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
)
_DATE = (
    r"(?:\d{1,2}[./-]\d{1,2}[./-](?:19|20)?\d{2}"
    r"|(?:19|20)\d{2}-\d{1,2}-\d{1,2}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\.?\s+(?:de\s+)?(?:{_MONTHS})\.?,?\s+(?:de\s+)?(?:19|20)\d{{2}}(?:\s*г\.?)?"
    rf"|(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}})"
)
_DOB = re.compile(
    r"(?:\bdob\b|\bd\.o\.b\.?|date of birth|birth ?date|\bborn(?: on| in)?\b|birthday|\bb\.\s?|"
    r"geboren(?: am| op)?|geburtsdatum|geboortedatum|geb\.|fecha de nacimiento|naci(?:do|da|ó) el|f\. nac\.?|"
    r"дата рождения|д\.\s?р\.|родил(?:ся|ась))"
    rf"[^\n\d]{{0,25}}?({_DATE})",
    re.IGNORECASE,
)


@dataclass
class DateOfBirthRecognizer:
    name: str = "date_of_birth"

    def find(self, text: str) -> list[Span]:
        return [
            _span(text, m.start(1), m.end(1), EntityType.DATE_OF_BIRTH, self.name, 0.9, validated=False)
            for m in _DOB.finditer(text)
        ]


# ------------------------------------------------------------------------------------------- secrets
_KNOWN_SECRETS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        ("openai_or_anthropic", r"\bsk-(?:proj-|ant-[a-z0-9]+-|or-v1-|svcacct-)?[A-Za-z0-9_-]{20,}"),
        ("github", r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
        ("gitlab", r"\bglpat-[A-Za-z0-9_-]{20,}"),
        ("slack", r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
        ("aws_access_key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        ("google_api_key", r"\bAIza[0-9A-Za-z_-]{35}"),
        ("stripe", r"\b[spr]k_(?:live|test)_[A-Za-z0-9]{16,}"),
        ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
        ("private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
    )
)
_SECRET_KEYWORDS = (
    r"password|passwd|pwd|passcode|pass|secret|api[_ -]?key|apikey|access[_ -]?key|token|auth|bearer|credential|"
    r"client[_ -]?secret|private[_ -]?key|passwort|kennwort|contraseña|clave|пароль|ключ|токен"
)
_ASSIGNMENT = re.compile(
    rf"(?i:\b(?:{_SECRET_KEYWORDS})\b[\w-]*)[\"']?"
    r"(?:\s*(?:[:=]|=>)|\s+(?:is|ist|es|era|was|lautet|-)\s|[^\n:=\"]{1,30}?:)\s*[\"']?"
    r"([^\s\"',;<>]{6,})"
)
# "my password Tulip!Garden2023": keyword, whitespace, then something that looks like a password
_BARE_PASSWORD = re.compile(r"(?i:\b(?:password|passwort|contraseña|пароль|passcode|pin)\b)\s+([^\s\"',;<>]{8,})")
_BEARER = re.compile(r"(?i:\bbearer)\s+([A-Za-z0-9._~+/-]{16,}=*)")
_CONNECTION = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:([^\s@/]+)@", re.IGNORECASE)
_TOKEN = re.compile(r"(?<![\w/+=-])[A-Za-z0-9_\-+/]{12,}={0,2}(?![\w/+=-])")
_SECRET_CONTEXT = re.compile(rf"(?i:{_SECRET_KEYWORDS})")


@dataclass
class SecretRecognizer:
    name: str = "secret"

    def find(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for name, pattern in _KNOWN_SECRETS:
            for m in pattern.finditer(text):
                spans.append(_span(text, m.start(), m.end(), EntityType.SECRET, f"secret_{name}", 0.99, validated=True))
        for pattern, source in ((_BEARER, "secret_bearer"), (_CONNECTION, "secret_connection_string")):
            for m in pattern.finditer(text):
                spans.append(_span(text, m.start(1), m.end(1), EntityType.SECRET, source, 0.95, validated=False))
        for m in _ASSIGNMENT.finditer(text):
            value = m.group(1)
            if v.looks_like_secret(value, with_context=True) or (len(value) >= 8 and re.search(r"\d", value)):
                spans.append(
                    _span(text, m.start(1), m.end(1), EntityType.SECRET, "secret_assignment", 0.9, validated=False)
                )
        for m in _BARE_PASSWORD.finditer(text):
            value = m.group(1)
            if (
                re.search(r"[A-Za-z]", value)
                and re.search(r"[\d\W_]", value)
                and v.looks_like_secret(value, with_context=True)
            ):
                spans.append(
                    _span(text, m.start(1), m.end(1), EntityType.SECRET, "secret_password", 0.85, validated=False)
                )
        for m in _TOKEN.finditer(text):
            with_context = bool(_SECRET_CONTEXT.search(_before(text, m.start(), 30)))
            if v.looks_like_secret(m.group(0), with_context=with_context):
                score = 0.85 if with_context else 0.7
                spans.append(
                    _span(text, m.start(), m.end(), EntityType.SECRET, "secret_entropy", score, validated=False)
                )
        return spans


# ------------------------------------------------------------------------------------- registration
@dataclass
class PatternDetector:
    """Runs every recognizer; overlapping candidates are resolved later by `merge.resolve_overlaps`."""

    recognizers: list[Recognizer] = field(default_factory=list)
    name: str = "patterns"

    @classmethod
    def default(cls, phone_regions: Iterable[str] = DEFAULT_PHONE_REGIONS) -> PatternDetector:
        return cls(
            recognizers=[
                EmailRecognizer(),
                PhoneRecognizer(regions=tuple(phone_regions)),
                CreditCardRecognizer(),
                IbanRecognizer(),
                IpAddressRecognizer(),
                UrlRecognizer(),
                UsSsnRecognizer(),
                NationalIdRecognizer(),
                DateOfBirthRecognizer(),
                SecretRecognizer(),
            ]
        )

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for recognizer in self.recognizers:
            spans.extend(recognizer.find(text))
        return spans
