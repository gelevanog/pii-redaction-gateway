"""Shared fixtures. Nothing here needs an API key or a model download."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field

import httpx
import pytest
from fastapi import FastAPI

from pii_shield.config import Settings
from pii_shield.detect.pipeline import DetectionPipeline
from pii_shield.entities import EntityType, Span
from pii_shield.gateway.app import create_app
from pii_shield.gateway.runtime import Runtime, build_runtime
from pii_shield.policy import Policy, PolicySet
from pii_shield.providers.fake import FakeProvider, RecordingProvider
from pii_shield.shield import PACKAGED_POLICIES, Shield
from pii_shield.vault.store import MemoryBackend, Vault

TEST_KEY = b"k" * 32


@dataclass
class StubNer:
    """A tiny stand-in for GLiNER: finds a fixed list of names, addresses and organizations."""

    people: list[str] = field(default_factory=lambda: ["Anna Petrova", "David Okafor", "Marcus Webb", "Grace Hall"])
    addresses: list[str] = field(default_factory=lambda: ["14 Elm Grove, Bristol BS6 5NP"])
    organizations: list[str] = field(default_factory=lambda: ["Acme Logistics"])
    name: str = "ner"

    def detect(self, text: str) -> list[Span]:
        spans = []
        for kind, values in (
            (EntityType.PERSON, self.people),
            (EntityType.ADDRESS, self.addresses),
            (EntityType.ORGANIZATION, self.organizations),
        ):
            for value in values:
                for match in re.finditer(re.escape(value), text):
                    spans.append(
                        Span(start=match.start(), end=match.end(), type=kind, text=value, score=0.9, source="ner")
                    )
        return spans


@pytest.fixture
def stub_ner() -> StubNer:
    return StubNer()


@pytest.fixture
def policies() -> PolicySet:
    return PolicySet.from_dir(PACKAGED_POLICIES, "support-chat")


@pytest.fixture
def support_policy(policies: PolicySet) -> Policy:
    return policies.get("support-chat")


@pytest.fixture
def pipeline(stub_ner: StubNer) -> DetectionPipeline:
    return DetectionPipeline(ner=stub_ner, ner_expected=True)


@pytest.fixture
def shield(policies: PolicySet, pipeline: DetectionPipeline) -> Shield:
    return Shield(policies=policies, pipeline=pipeline, vault=Vault(MemoryBackend(), TEST_KEY), secret=b"s" * 32)


@pytest.fixture
def settings() -> Settings:
    return Settings(ner_enabled=False, openrouter_api_key=None, audit_file=None, tenants_file=None, _env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def recorder() -> RecordingProvider:
    return RecordingProvider(FakeProvider())


@pytest.fixture
def runtime(settings: Settings, recorder: RecordingProvider, pipeline: DetectionPipeline) -> Runtime:
    return build_runtime(settings, upstream=recorder, pipeline=pipeline)


@pytest.fixture
def app(settings: Settings, runtime: Runtime) -> FastAPI:
    return create_app(settings, runtime)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as http:
        yield http


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PII_SHIELD_VAULT_KEY"):
        monkeypatch.delenv(name, raising=False)
    yield
