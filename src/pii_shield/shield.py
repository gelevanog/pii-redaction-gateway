"""The library facade: `shield.redact(text)` and `shield.restore(text, session)`.

    shield = Shield.create(policy="support-chat")
    result = shield.redact("Hi, I'm Anna Petrova, anna.petrova@gmail.com")
    result.text            # "Hi, I'm <PERSON_1>, <EMAIL_1>"
    shield.restore("Dear <PERSON_1>, ...", result)   # "Dear Anna Petrova, ..."

For several messages of one conversation use a session, so placeholders stay consistent and the vault
is locked, loaded and saved once:

    with shield.session("conversation-42") as s:
        first = s.redact(message_1)
        second = s.redact(message_2)
        answer = s.restore(llm_answer)
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, Field

from pii_shield.anonymize.restore import RestoreReport, StreamRestorer, restore_text
from pii_shield.anonymize.strategies import Anonymizer
from pii_shield.detect.pipeline import DetectionPipeline
from pii_shield.entities import NER_TYPES, EntityType
from pii_shield.policy import Action, Policy, PolicySet, load_policy
from pii_shield.vault.crypto import decode_key
from pii_shield.vault.session import SessionState
from pii_shield.vault.store import MemoryBackend, Vault

PACKAGED_POLICIES = Path(__file__).parent / "policies"


class RedactedEntity(BaseModel):
    type: EntityType
    start: int
    end: int
    score: float
    source: str
    action: Action
    replacement: str
    original: str = Field(repr=False)
    """The raw value. Present for in-process callers (highlighting); never written to logs or the audit log."""


class RedactionResult(BaseModel):
    text: str
    """The text to send to the LLM."""
    session_id: str
    policy: str
    entities: list[RedactedEntity] = Field(default_factory=list)
    blocked: bool = False
    block_reasons: list[str] = Field(default_factory=list)
    detector_errors: dict[str, str] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return dict(Counter(entity.type.value for entity in self.entities))

    def action_counts(self) -> dict[str, int]:
        return dict(Counter(entity.action.value for entity in self.entities))


class BlockedError(RuntimeError):
    """Raised by `redact_or_raise` when the policy refuses the input."""

    def __init__(self, result: RedactionResult) -> None:
        super().__init__("request blocked by PII policy: " + ", ".join(result.block_reasons))
        self.result = result


class ShieldSession:
    """One conversation: a locked, loaded vault state plus the policy to apply."""

    def __init__(self, shield: Shield, state: SessionState, policy: Policy) -> None:
        self.shield = shield
        self.state = state
        self.policy = policy

    @property
    def session_id(self) -> str:
        return self.state.session_id

    def redact(self, text: str) -> RedactionResult:
        started = time.perf_counter()
        people = [e.original for e in self.state.entries if e.type is EntityType.PERSON]
        values = [(e.type, e.original) for e in self.state.entries if e.type is not EntityType.PERSON]
        outcome = self.shield.pipeline.detect(text, self.policy, known_people=people, known_values=values)
        result = RedactionResult(
            text=text,
            session_id=self.session_id,
            policy=self.policy.name,
            detector_errors=outcome.errors,
            timings_ms=dict(outcome.timings_ms),
        )
        pieces: list[str] = []
        cursor = 0
        for span in outcome.spans:
            rule = self.policy.rule(span.type)
            replacement = self.shield.anonymizer.replace(
                self.state, span.type, span.text, rule, link_names=self.policy.link_name_variants
            )
            pieces += [text[cursor : span.start], replacement]
            cursor = span.end
            result.entities.append(
                RedactedEntity(
                    type=span.type,
                    start=span.start,
                    end=span.end,
                    score=span.score,
                    source=span.source,
                    action=rule.action,
                    replacement=replacement,
                    original=span.text,
                )
            )
            if rule.action is Action.BLOCK and span.type.value not in result.block_reasons:
                result.block_reasons.append(span.type.value)
        pieces.append(text[cursor:])
        result.text = "".join(pieces)
        if self.policy.fail_closed:
            for detector in self._required_failures(outcome.errors):
                result.block_reasons.append(f"detector_unavailable:{detector}")
        result.blocked = bool(result.block_reasons)
        result.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    def _required_failures(self, errors: dict[str, str]) -> list[str]:
        failed = []
        for detector in errors:
            if detector == "ner" and not (NER_TYPES & self.policy.protected_types):
                continue  # the policy keeps names/addresses/orgs anyway
            failed.append(detector)
        return failed

    def restore(self, text: str) -> str:
        return restore_text(text, self.state)[0]

    def restore_with_report(self, text: str, *, json_string: bool = False) -> tuple[str, RestoreReport]:
        return restore_text(text, self.state, json_string=json_string)

    def stream_restorer(self, *, json_string: bool = False) -> StreamRestorer:
        return StreamRestorer(self.state, json_string=json_string)


class Shield:
    def __init__(self, *, policies: PolicySet, pipeline: DetectionPipeline, vault: Vault, secret: bytes) -> None:
        self.policies = policies
        self.pipeline = pipeline
        self.vault = vault
        self.anonymizer = Anonymizer(secret)

    @classmethod
    def create(
        cls,
        policy: str | Path | Policy = "support-chat",
        *,
        ner: bool | str = False,
        vault: Vault | None = None,
        vault_key: str | None = None,
        ttl_seconds: int = 3600,
    ) -> Shield:
        """Convenience constructor for library use.

        `policy` is a packaged policy name, a YAML path or a `Policy`. `ner=True` loads the default GLiNER
        model (needs the `ner` extra), a string picks another model id.
        """
        if isinstance(policy, Policy):
            loaded = policy
        elif isinstance(policy, Path) or str(policy).endswith((".yaml", ".yml")):
            loaded = load_policy(policy)
        else:
            loaded = load_policy(PACKAGED_POLICIES / f"{policy}.yaml")
        others = [load_policy(p) for p in sorted(PACKAGED_POLICIES.glob("*.yaml")) if p.stem != loaded.name]
        policies = PolicySet([loaded, *others], default=loaded.name)
        pipeline = DetectionPipeline(ner_expected=bool(ner))
        if ner:
            from pii_shield.detect.ner import DEFAULT_NER_MODEL, GlinerDetector

            pipeline.ner = GlinerDetector(model_name=ner if isinstance(ner, str) else DEFAULT_NER_MODEL)
        key = decode_key(vault_key) if vault_key else secrets.token_bytes(32)
        vault = vault or Vault(MemoryBackend(), key, ttl_seconds=ttl_seconds)
        return cls(policies=policies, pipeline=pipeline, vault=vault, secret=derive_secret(key))

    @contextmanager
    def session(self, session_id: str | None = None, policy: str | None = None) -> Iterator[ShieldSession]:
        session_id = session_id or new_session_id()
        with self.vault.session(session_id) as state:
            yield ShieldSession(self, state, self.policies.get(policy))

    def redact(self, text: str, *, policy: str | None = None, session_id: str | None = None) -> RedactionResult:
        with self.session(session_id, policy) as session:
            return session.redact(text)

    def redact_or_raise(
        self, text: str, *, policy: str | None = None, session_id: str | None = None
    ) -> RedactionResult:
        result = self.redact(text, policy=policy, session_id=session_id)
        if result.blocked:
            raise BlockedError(result)
        return result

    def restore(self, text: str, session: str | RedactionResult) -> str:
        session_id = session.session_id if isinstance(session, RedactionResult) else session
        with self.session(session_id) as s:
            return s.restore(text)


def new_session_id() -> str:
    return uuid.uuid4().hex


def derive_secret(vault_key: bytes, hash_key: str | None = None) -> bytes:
    """Key for the hash action and synthetic seeds: explicit, or derived from the vault key."""
    if hash_key:
        return hashlib.sha256(b"pii-shield:hash:" + hash_key.encode("utf-8")).digest()
    return hashlib.sha256(b"pii-shield:hash:" + base64.b64encode(vault_key)).digest()
