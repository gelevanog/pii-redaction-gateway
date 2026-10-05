"""Turn one detected span into its replacement according to the policy action."""

from __future__ import annotations

import hashlib
import hmac

from pii_shield.anonymize.normalize import name_tokens, normalize
from pii_shield.anonymize.synthetic import seed_for, synthesize
from pii_shield.entities import EntityType
from pii_shield.policy import Action, EntityRule
from pii_shield.vault.session import SessionState, VaultEntry


def mask(value: str, keep_last: int = 0, char: str = "*") -> str:
    """Replace letters and digits with `char`, keep separators and the last `keep_last` alphanumerics."""
    alnum_positions = [i for i, c in enumerate(value) if c.isalnum()]
    visible = set(alnum_positions[-keep_last:]) if keep_last else set()
    return "".join(c if (not c.isalnum() or i in visible) else char for i, c in enumerate(value))


def keyed_hash(kind: EntityType, value: str, key: bytes, length: int = 8) -> str:
    """Irreversible, consistent token: same value -> same token for everyone holding the key, no lookup."""
    digest = hmac.new(key, f"{kind.value}\x1f{normalize(kind, value)}".encode(), hashlib.sha256).hexdigest()
    return f"<{kind.value}:{digest[:length]}>"


class Anonymizer:
    """Applies actions; reversible ones go through the session so they can be restored later."""

    def __init__(self, secret: bytes) -> None:
        self._secret = secret

    def replace(
        self,
        state: SessionState,
        kind: EntityType,
        value: str,
        rule: EntityRule,
        *,
        link_names: bool = True,
    ) -> str:
        match rule.action:
            case Action.KEEP:
                return value
            case Action.MASK:
                return mask(value, rule.keep_last)
            case Action.HASH:
                return keyed_hash(kind, value, self._secret)
            case Action.BLOCK:
                # The request will be refused; the value is still masked in case the caller logs the result.
                return f"[{kind.value} BLOCKED]"
            case Action.PSEUDONYMIZE:
                entry = state.get_or_create(kind, value, link_names=link_names)
                return state.assign_placeholder(entry)
            case Action.SYNTHETIC:
                if kind is EntityType.CUSTOM:
                    entry = state.get_or_create(kind, value, link_names=link_names)
                    return state.assign_placeholder(entry)
                entry = state.get_or_create(kind, value, link_names=link_names)
                surface = self._synthetic_surface(state, entry, kind, value)
                state.record_synthetic(entry, surface, value)
                return surface

    def _synthetic_surface(self, state: SessionState, entry: VaultEntry, kind: EntityType, value: str) -> str:
        if entry.synthetic is None:
            seed = seed_for(self._secret, state.session_id, kind.value, entry.key)
            entry.synthetic = synthesize(kind, entry.original, seed)
            state.dirty = True
        if kind is not EntityType.PERSON:
            return entry.synthetic
        # Name variants map position-wise onto the synthetic name: "Petrova" -> fake last name.
        original_tokens = name_tokens(entry.original)
        fake_tokens = entry.synthetic.split()
        tokens = name_tokens(value)
        if tokens == original_tokens or len(fake_tokens) < 2:
            return entry.synthetic if tokens == original_tokens else fake_tokens[0]
        if len(tokens) == 1 and original_tokens:
            if tokens[0] == original_tokens[0]:
                return fake_tokens[0]
            if tokens[0] == original_tokens[-1]:
                return fake_tokens[-1]
        return entry.synthetic
