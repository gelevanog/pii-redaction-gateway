"""Wire settings into a running shield: policies, detectors, vault, upstream provider, audit log, tenants."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from pii_shield.audit import AuditLog
from pii_shield.config import Settings
from pii_shield.detect.llm import LlmDetector
from pii_shield.detect.ner import GlinerDetector, NerUnavailableError, ner_installed
from pii_shield.detect.pipeline import DetectionPipeline
from pii_shield.logging_config import get_logger
from pii_shield.policy import PolicySet
from pii_shield.providers.base import ChatProvider, ProviderError
from pii_shield.providers.factory import budgeted, build_provider
from pii_shield.providers.fake import FakeProvider
from pii_shield.shield import Shield, derive_secret
from pii_shield.vault.crypto import decode_key
from pii_shield.vault.store import FileBackend, MemoryBackend, RedisBackend, Vault, VaultBackend

log = get_logger(__name__)


class Tenant(BaseModel):
    name: str
    api_key_sha256: str
    policy: str
    allowed_policies: list[str] = Field(default_factory=list)
    """Policies this tenant may pick with the X-PII-Policy header or a /p/<policy>/ route."""


class TenantRegistry:
    """API key (by SHA-256) -> tenant. Without a tenants file the gateway is open (local development)."""

    def __init__(self, tenants: list[Tenant]) -> None:
        self._by_hash = {tenant.api_key_sha256.lower(): tenant for tenant in tenants}

    @classmethod
    def from_file(cls, path: Path | None) -> TenantRegistry:
        if path is None:
            return cls([])
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls([Tenant.model_validate(item) for item in raw.get("tenants", [])])

    @property
    def enabled(self) -> bool:
        return bool(self._by_hash)

    def lookup(self, api_key: str | None) -> Tenant | None:
        if not api_key:
            return None
        return self._by_hash.get(hashlib.sha256(api_key.encode("utf-8")).hexdigest())


@dataclass
class Runtime:
    settings: Settings
    shield: Shield
    upstream: ChatProvider
    audit: AuditLog
    tenants: TenantRegistry
    playground_providers: dict[str, ChatProvider] = field(default_factory=dict)
    ner_status: str = "disabled"
    ephemeral_key: bool = False


def build_vault(settings: Settings) -> tuple[Vault, bytes, bool]:
    ephemeral = settings.vault_key is None
    key = decode_key(settings.vault_key) if settings.vault_key else secrets.token_bytes(32)
    backend: VaultBackend
    if settings.vault_backend == "redis":
        backend = RedisBackend(settings.redis_url)
    elif settings.vault_backend == "file":
        backend = FileBackend(settings.vault_dir)
    else:
        backend = MemoryBackend()
    if ephemeral:
        log.warning("vault.ephemeral_key", hint="set PII_SHIELD_VAULT_KEY; sessions and hashes change on restart")
    return Vault(backend, key, ttl_seconds=settings.vault_ttl_seconds), key, ephemeral


def build_pipeline(settings: Settings) -> tuple[DetectionPipeline, str]:
    pipeline = DetectionPipeline(ner_expected=settings.ner_enabled)
    status = "disabled (patterns only)"
    if settings.ner_enabled:
        if not ner_installed():
            status = 'not installed (pip install "pii-shield[ner]")'
            log.warning("ner.not_installed")
        else:
            detector = GlinerDetector(model_name=settings.ner_model, threads=settings.ner_threads)
            pipeline.ner = detector
            status = f"{settings.ner_model} (loads on first use)"
            if settings.ner_preload:
                try:
                    detector.load()
                    status = f"{settings.ner_model} (loaded in {detector.load_seconds}s)"
                except NerUnavailableError as exc:
                    pipeline.ner = None
                    status = f"unavailable: {exc}"
                    log.error("ner.unavailable", error=str(exc)[:200])
    if settings.has_key(settings.llm_detector_provider):
        try:
            provider = budgeted(
                build_provider(
                    settings,
                    kind=settings.llm_detector_provider,
                    model=settings.llm_detector_model,
                    fallback_models=settings.llm_detector_fallback_models,
                ),
                settings,
                tag="llm_detector",
            )
            pipeline.llm = LlmDetector(
                provider=provider,
                model=settings.llm_detector_model,
                openrouter=settings.llm_detector_provider == "openrouter",
            )
        except ProviderError as exc:
            log.warning("llm_detector.unavailable", error=str(exc)[:200])
    return pipeline, status


def build_runtime(
    settings: Settings, *, upstream: ChatProvider | None = None, pipeline: DetectionPipeline | None = None
) -> Runtime:
    policies = PolicySet.from_dir(settings.policies_dir, settings.default_policy)
    vault, key, ephemeral = build_vault(settings)
    ner_status = "provided by caller"
    if pipeline is None:
        pipeline, ner_status = build_pipeline(settings)
    secret = derive_secret(key, settings.hash_key)
    shield = Shield(policies=policies, pipeline=pipeline, vault=vault, secret=secret)
    pipeline.patterns.detect("warm up +1 415 555 2671")  # loads phone metadata before the first request
    upstream = upstream or build_provider(settings)
    playground: dict[str, ChatProvider] = {"fake": FakeProvider()}
    if not isinstance(upstream, FakeProvider):
        playground["real"] = upstream
    elif settings.openrouter_api_key:
        try:
            playground["real"] = budgeted(build_provider(settings, kind="openrouter"), settings, tag="playground")
        except ProviderError as exc:
            log.warning("playground.real_unavailable", error=str(exc)[:200])
    audit = AuditLog(
        hashlib.sha256(b"pii-shield:audit:" + secret).digest(), settings.audit_max_entries, settings.audit_file
    )
    if settings.audit_file is not None:
        audit.preload(AuditLog.load_jsonl(settings.audit_file)[-settings.audit_max_entries :])
    return Runtime(
        settings=settings,
        shield=shield,
        upstream=upstream,
        audit=audit,
        tenants=TenantRegistry.from_file(settings.tenants_file),
        playground_providers=playground,
        ner_status=ner_status,
        ephemeral_key=ephemeral,
    )
