"""Entity types and detected spans: the vocabulary shared by detectors, policies, the vault and the eval."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EntityType(StrEnum):
    PERSON = "PERSON"
    ADDRESS = "ADDRESS"
    ORGANIZATION = "ORGANIZATION"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    IBAN = "IBAN"
    IP_ADDRESS = "IP_ADDRESS"
    URL = "URL"
    US_SSN = "US_SSN"
    NATIONAL_ID = "NATIONAL_ID"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    SECRET = "SECRET"
    CUSTOM = "CUSTOM"


# Types found by the NER model (names, places, organizations); everything else is pattern-based.
NER_TYPES: frozenset[EntityType] = frozenset({EntityType.PERSON, EntityType.ADDRESS, EntityType.ORGANIZATION})


class Span(BaseModel):
    """One detected entity: half-open character range [start, end) in the original text."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    type: EntityType
    text: str
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "pattern"
    """Which detector produced the span: pattern name, "ner", "llm" or "deny_list"."""
    validated: bool = False
    """True when a checksum or library validator confirmed the value (Luhn, mod-97, phonenumbers, ...)."""

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end
