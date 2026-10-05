"""PII Shield: strip personal data before it reaches an LLM, and put it back in the answer."""

from __future__ import annotations

__version__ = "0.1.0"

from pii_shield.entities import EntityType, Span
from pii_shield.policy import Action, Policy, load_policy
from pii_shield.shield import BlockedError, RedactionResult, Shield

__all__ = [
    "Action",
    "BlockedError",
    "EntityType",
    "Policy",
    "RedactionResult",
    "Shield",
    "Span",
    "__version__",
    "load_policy",
]
