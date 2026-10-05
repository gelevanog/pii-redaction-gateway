"""Put the original values back into an LLM answer, also when it arrives as a token stream.

Models are not perfectly faithful with placeholders: they may write `<PERSON_1>`, `[PERSON_1]`,
`{PERSON_1}`, `< PERSON_1 >`, `&lt;PERSON_1&gt;`, `<person_1>` or a bare `PERSON_1`. All of these are
restored when the number exists in the session. Placeholders the session does not know are left as they
are and reported, so a client can tell a hallucinated `<PERSON_7>` from a real one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pii_shield.entities import EntityType
from pii_shield.vault.session import SessionState

_TYPES = "|".join(sorted((t.value for t in EntityType), key=len, reverse=True))
PLACEHOLDER_PATTERN = re.compile(
    rf"(?i:(?:<|&lt;|\[|\{{)\s*({_TYPES})_(\d+)\s*(?:>|&gt;|\]|\}}))"  # bracketed, any bracket style or case
    rf"|(?<![A-Za-z0-9_])({_TYPES})_(\d+)(?![A-Za-z0-9_])"  # bare PERSON_1 (upper case only: not code identifiers)
)
_MAX_PLACEHOLDER = max(len(t.value) for t in EntityType) + 16  # "&lt;" + TYPE + "_" + digits + "&gt;"
_OPENERS = ("<", "&lt;", "[", "{")


@dataclass
class RestoreReport:
    restored: int = 0
    unknown: list[str] = field(default_factory=list)
    synthetic_restored: int = 0


def _escape_json(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)[1:-1]


def restore_text(text: str, state: SessionState, *, json_string: bool = False) -> tuple[str, RestoreReport]:
    """Replace placeholders (and synthetic stand-ins) with the originals from the session."""
    report = RestoreReport()

    def substitute(match: re.Match[str]) -> str:
        kind = (match.group(1) or match.group(3)).upper()
        number = match.group(2) or match.group(4)
        entry = state.by_placeholder(f"<{kind}_{int(number)}>")
        if entry is None:
            report.unknown.append(match.group(0))
            return match.group(0)
        report.restored += 1
        return _escape_json(entry.original) if json_string else entry.original

    restored = PLACEHOLDER_PATTERN.sub(substitute, text)
    synthetic = state.synthetic_map()
    if synthetic:
        pattern = re.compile("|".join(re.escape(s) for s in sorted(synthetic, key=len, reverse=True)))

        def substitute_synthetic(match: re.Match[str]) -> str:
            report.synthetic_restored += 1
            original = synthetic[match.group(0)]
            return _escape_json(original) if json_string else original

        restored = pattern.sub(substitute_synthetic, restored)
    return restored, report


def _could_be_placeholder_prefix(tail: str) -> bool:
    """True if `tail` might grow into a placeholder with more characters ("<PER", "PERSON_1", "[EMAIL_")."""
    if not tail:
        return False
    body = tail
    for opener in _OPENERS:
        if body.lower().startswith(opener):
            body = body[len(opener) :].lstrip()
            if not body:
                return True
            break
        if opener.startswith(body.lower()):
            return True  # "&l" could become "&lt;"
    upper = body.upper()
    for kind in EntityType:
        name = kind.value + "_"
        if name.startswith(upper):
            return True
        if upper.startswith(name):
            rest = upper[len(name) :]
            digits = len(rest) - len(rest.lstrip("0123456789"))
            remainder = rest[digits:].strip()
            if digits and remainder in {"", "&", "&G", "&GT"}:
                return True  # "PERSON_1" may still become "PERSON_12" or get its closing bracket
            if not rest:
                return True
    return False


class StreamRestorer:
    """Incremental restore for streamed text: emits everything except a tail that may be a split placeholder.

    Feed chunks with `push()`, call `flush()` at the end of the stream. The held-back tail is at most
    ~40 characters, so the added latency is at most one chunk.
    """

    def __init__(self, state: SessionState, *, json_string: bool = False) -> None:
        self._state = state
        self._json = json_string
        self._buffer = ""
        self.report = RestoreReport()
        self._synthetic = state.synthetic_map()
        self._hold = max([_MAX_PLACEHOLDER, *(len(s) for s in self._synthetic)])

    def _safe_cut(self, text: str) -> int:
        """Index up to which `text` can be emitted without cutting a placeholder in half."""
        window_start = max(0, len(text) - self._hold)
        for index in range(window_start, len(text)):
            char = text[index]
            starts_token = char in "<&[{" or (
                char.isalpha() and (index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_"))
            )
            if starts_token and _could_be_placeholder_prefix(text[index:]):
                return index
        if self._synthetic:
            # A synthetic value could also be split: hold back any suffix that is a prefix of one.
            for index in range(window_start, len(text)):
                tail = text[index:]
                if any(s.startswith(tail) for s in self._synthetic):
                    return index
        return len(text)

    def push(self, chunk: str) -> str:
        self._buffer += chunk
        cut = self._safe_cut(self._buffer)
        ready, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return self._restore(ready)

    def flush(self) -> str:
        ready, self._buffer = self._buffer, ""
        return self._restore(ready)

    def _restore(self, text: str) -> str:
        if not text:
            return ""
        restored, report = restore_text(text, self._state, json_string=self._json)
        self.report.restored += report.restored
        self.report.synthetic_restored += report.synthetic_restored
        self.report.unknown.extend(report.unknown)
        return restored
