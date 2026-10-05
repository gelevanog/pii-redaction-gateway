"""Per-conversation state: which original value got which placeholder or synthetic stand-in.

Persisted (encrypted) by `Vault`. The same value always gets the same placeholder within a session,
and a person mentioned as "Anna Petrova", "Ms. Petrova" and "Anna" is one `<PERSON_1>` when the link
is unambiguous.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, PrivateAttr

from pii_shield.anonymize.normalize import name_tokens, normalize
from pii_shield.entities import EntityType


class VaultEntry(BaseModel):
    type: EntityType
    key: str
    """Normalized value (see `normalize`); the lookup key within the session."""
    original: str
    """Canonical surface form used when restoring (for people: the longest form seen)."""
    placeholder: str | None = None
    synthetic: str | None = None
    tokens: list[str] = Field(default_factory=list)
    """Name tokens, for linking name variants of the same person."""
    synthetic_variants: dict[str, str] = Field(default_factory=dict)
    """Synthetic surface -> original surface, for every variant replaced with a synthetic value."""


class SessionState(BaseModel):
    session_id: str
    entries: list[VaultEntry] = Field(default_factory=list)
    counters: dict[str, int] = Field(default_factory=dict)
    dirty: bool = Field(default=False, exclude=True)
    _by_key: dict[tuple[EntityType, str], VaultEntry] = PrivateAttr(default_factory=dict)
    _by_placeholder: dict[str, VaultEntry] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        for entry in self.entries:
            self._index(entry)

    def _index(self, entry: VaultEntry) -> None:
        self._by_key[(entry.type, entry.key)] = entry
        if entry.placeholder:
            self._by_placeholder[entry.placeholder] = entry

    # ---------------------------------------------------------------------------------- lookups
    def find(self, kind: EntityType, value: str, *, link_names: bool = True) -> VaultEntry | None:
        key = normalize(kind, value)
        entry = self._by_key.get((kind, key))
        if entry is not None or kind is not EntityType.PERSON or not link_names:
            return entry
        return self._link_person(name_tokens(value))

    def _link_person(self, tokens: list[str]) -> VaultEntry | None:
        """Unambiguous subset/superset match against known people ("Anna" -> "Anna Petrova")."""
        if not tokens:
            return None
        new = set(tokens)
        candidates = [
            entry
            for entry in self.entries
            if entry.type is EntityType.PERSON
            and entry.tokens
            and (new <= set(entry.tokens) or set(entry.tokens) <= new)
        ]
        if len(candidates) != 1:
            return None  # unknown or ambiguous ("Anna" with two Annas): a separate placeholder is safer
        return candidates[0]

    def by_placeholder(self, placeholder: str) -> VaultEntry | None:
        return self._by_placeholder.get(placeholder)

    @property
    def placeholders(self) -> dict[str, VaultEntry]:
        return dict(self._by_placeholder)

    def synthetic_map(self) -> dict[str, str]:
        """Synthetic surface -> original surface over the whole session (longest first is the caller's job)."""
        mapping: dict[str, str] = {}
        for entry in self.entries:
            mapping.update(entry.synthetic_variants)
        return mapping

    # ---------------------------------------------------------------------------------- updates
    def get_or_create(self, kind: EntityType, value: str, *, link_names: bool = True) -> VaultEntry:
        entry = self.find(kind, value, link_names=link_names)
        if entry is None:
            entry = VaultEntry(
                type=kind,
                key=normalize(kind, value),
                original=value,
                tokens=name_tokens(value) if kind is EntityType.PERSON else [],
            )
            self.entries.append(entry)
            self._index(entry)
            self.dirty = True
        elif kind is EntityType.PERSON:
            self._absorb_variant(entry, value)
        return entry

    def _absorb_variant(self, entry: VaultEntry, value: str) -> None:
        tokens = name_tokens(value)
        merged = list(dict.fromkeys([*entry.tokens, *tokens]))
        if merged != entry.tokens:
            entry.tokens = merged
            self.dirty = True
        if len(tokens) > len(name_tokens(entry.original)):
            entry.original = value  # restore to the fullest form seen ("Anna" -> "Anna Petrova")
            self.dirty = True
        key = normalize(EntityType.PERSON, value)
        if (EntityType.PERSON, key) not in self._by_key:
            self._by_key[(EntityType.PERSON, key)] = entry

    def assign_placeholder(self, entry: VaultEntry) -> str:
        if entry.placeholder is None:
            number = self.counters.get(entry.type.value, 0) + 1
            self.counters[entry.type.value] = number
            entry.placeholder = f"<{entry.type.value}_{number}>"
            self._by_placeholder[entry.placeholder] = entry
            self.dirty = True
        return entry.placeholder

    def record_synthetic(self, entry: VaultEntry, synthetic_surface: str, original_surface: str) -> None:
        if entry.synthetic is None:
            entry.synthetic = synthetic_surface
        if entry.synthetic_variants.get(synthetic_surface) != original_surface:
            entry.synthetic_variants[synthetic_surface] = original_surface
            self.dirty = True
