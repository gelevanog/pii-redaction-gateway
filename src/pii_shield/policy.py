"""Redaction policies: which entities, which action, which thresholds, allow- and deny-lists.

Policies are YAML files (one per tenant or route); see configs/policies/ for the shipped examples.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pii_shield.entities import NER_TYPES, EntityType


class Action(StrEnum):
    PSEUDONYMIZE = "pseudonymize"
    """Reversible placeholder `<PERSON_1>`, restored in the answer."""
    SYNTHETIC = "synthetic"
    """Realistic fake value (Faker, format-preserving), reversible through the vault."""
    MASK = "mask"
    """Irreversible `****1234`."""
    HASH = "hash"
    """Irreversible but consistent keyed hash `<EMAIL:3f2a9c1b>` (joins in analytics, no lookup)."""
    BLOCK = "block"
    """Refuse the whole request."""
    KEEP = "keep"
    """Leave the value untouched (still counted in the audit log)."""


REVERSIBLE_ACTIONS = frozenset({Action.PSEUDONYMIZE, Action.SYNTHETIC})


class PolicyError(ValueError):
    pass


class EntityRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action = Action.PSEUDONYMIZE
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    keep_last: int = Field(default=0, ge=0, le=8)
    """For `mask`: characters left visible at the end (e.g. 4 for card numbers)."""


class AllowItem(BaseModel):
    """Never redact: an exact value (case-insensitive) or a full-match regex, optionally for one entity type."""

    model_config = ConfigDict(extra="forbid")

    value: str | None = None
    pattern: str | None = None
    entity: EntityType | None = None

    @model_validator(mode="after")
    def _one_of(self) -> Self:
        if (self.value is None) == (self.pattern is None):
            raise ValueError("allow-list item needs exactly one of `value` or `pattern`")
        if self.pattern is not None:
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(f"invalid allow-list pattern {self.pattern!r}: {exc}") from exc
        return self

    def matches(self, kind: EntityType, value: str) -> bool:
        if self.entity is not None and self.entity is not kind:
            return False
        if self.value is not None:
            return value.strip().casefold() == self.value.strip().casefold()
        assert self.pattern is not None
        return re.fullmatch(self.pattern, value.strip(), re.IGNORECASE) is not None


class DenyItem(BaseModel):
    """Always redact: a term (case-insensitive, whole word) or a regex, as the given entity type."""

    model_config = ConfigDict(extra="forbid")

    value: str | None = None
    pattern: str | None = None
    entity: EntityType = EntityType.CUSTOM

    @model_validator(mode="after")
    def _one_of(self) -> Self:
        if (self.value is None) == (self.pattern is None):
            raise ValueError("deny-list item needs exactly one of `value` or `pattern`")
        return self

    def regex(self) -> re.Pattern[str]:
        if self.value is not None:
            return re.compile(rf"(?<!\w){re.escape(self.value)}(?!\w)", re.IGNORECASE)
        assert self.pattern is not None
        return re.compile(self.pattern)


class DetectorToggles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patterns: bool = True
    ner: bool = True
    llm: bool = False


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    default_action: Action = Action.PSEUDONYMIZE
    """Action for entity types without their own rule."""
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    """Default confidence threshold; per-entity `threshold` overrides it."""
    entities: dict[EntityType, EntityRule] = Field(default_factory=dict)
    allow_list: list[AllowItem] = Field(default_factory=list)
    deny_list: list[DenyItem] = Field(default_factory=list)
    detectors: DetectorToggles = Field(default_factory=DetectorToggles)
    link_name_variants: bool = True
    fail_closed: bool = True
    """Refuse the request when a detector the policy relies on fails or is unavailable."""

    @field_validator("allow_list", mode="before")
    @classmethod
    def _allow_strings(cls, value: object) -> object:
        if isinstance(value, list):
            return [{"value": item} if isinstance(item, str) else item for item in value]
        return value

    @field_validator("deny_list", mode="before")
    @classmethod
    def _deny_strings(cls, value: object) -> object:
        if isinstance(value, list):
            return [{"value": item} if isinstance(item, str) else item for item in value]
        return value

    def rule(self, kind: EntityType) -> EntityRule:
        return self.entities.get(kind) or EntityRule(action=self.default_action)

    def action(self, kind: EntityType) -> Action:
        return self.rule(kind).action

    def threshold(self, kind: EntityType) -> float:
        rule = self.entities.get(kind)
        return rule.threshold if rule is not None and rule.threshold is not None else self.min_score

    def thresholds(self) -> dict[EntityType, float]:
        return {kind: self.threshold(kind) for kind in EntityType}

    def is_allowed(self, kind: EntityType, value: str) -> bool:
        return any(item.matches(kind, value) for item in self.allow_list)

    @property
    def protected_types(self) -> set[EntityType]:
        """Entity types this policy removes from the text (everything except `keep`)."""
        return {kind for kind in EntityType if self.action(kind) is not Action.KEEP}

    @property
    def blocking_types(self) -> set[EntityType]:
        return {kind for kind in EntityType if self.action(kind) is Action.BLOCK}

    @property
    def needs_ner(self) -> bool:
        return self.detectors.ner and bool(NER_TYPES & self.protected_types)


def load_policy(path: Path | str) -> Policy:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PolicyError(f"{path}: invalid YAML: {exc}") from exc
    raw.setdefault("name", path.stem)
    try:
        return Policy.model_validate(raw)
    except ValueError as exc:
        raise PolicyError(f"{path}: {exc}") from exc


class PolicySet:
    """All policies of a deployment, by name."""

    def __init__(self, policies: list[Policy], default: str) -> None:
        self._policies = {policy.name: policy for policy in policies}
        if default not in self._policies:
            raise PolicyError(f"default policy {default!r} not found (have: {', '.join(sorted(self._policies))})")
        self.default_name = default

    @classmethod
    def from_dir(cls, directory: Path | str, default: str) -> PolicySet:
        directory = Path(directory)
        files = sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")])
        if not files:
            raise PolicyError(f"no policy files in {directory}")
        return cls([load_policy(path) for path in files], default)

    def get(self, name: str | None = None) -> Policy:
        key = name or self.default_name
        try:
            return self._policies[key]
        except KeyError as exc:
            raise PolicyError(f"unknown policy {key!r} (have: {', '.join(sorted(self._policies))})") from exc

    @property
    def default(self) -> Policy:
        return self._policies[self.default_name]

    @property
    def names(self) -> list[str]:
        return sorted(self._policies)

    def __iter__(self) -> Iterator[Policy]:
        return iter(self._policies.values())
