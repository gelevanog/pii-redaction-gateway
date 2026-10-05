"""Runtime settings from environment variables (and an optional .env file). See .env.example."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from pii_shield.detect.ner import DEFAULT_NER_MODEL

ProviderKind = Literal["fake", "openrouter", "openai", "anthropic"]

DEFAULT_FREE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_FREE_FALLBACKS = ["google/gemma-4-31b-it:free", "dots-studio/dots-3-note-preview:free"]
DEFAULT_MODELS: dict[str, str] = {
    "fake": "fake-echo",
    "openrouter": DEFAULT_FREE_MODEL,
    "openai": "gpt-5-mini",
    "anthropic": "claude-sonnet-5",
}


def _split(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PII_SHIELD_", extra="ignore")

    # ---- upstream LLM
    upstream_provider: ProviderKind = "fake"
    upstream_model: str = ""
    """Default model when the client sends none (or "auto"); per provider default if empty."""
    upstream_fallback_models: Annotated[list[str], NoDecode] = Field(default_factory=list)
    require_free_models: bool = True
    """Refuse any OpenRouter model id that does not end in ":free" (requests and served models)."""
    upstream_timeout_seconds: float = 120.0

    openrouter_api_key: str | None = Field(default=None, validation_alias=AliasChoices("OPENROUTER_API_KEY"))
    openai_api_key: str | None = Field(default=None, validation_alias=AliasChoices("OPENAI_API_KEY"))
    anthropic_api_key: str | None = Field(default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY"))
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias=AliasChoices("OPENROUTER_BASE_URL")
    )
    openai_base_url: str = Field(default="https://api.openai.com/v1", validation_alias=AliasChoices("OPENAI_BASE_URL"))

    # ---- policies and tenants
    policies_dir: Path = Path(__file__).parent / "policies"
    default_policy: str = "support-chat"
    allow_policy_header: bool = True
    placeholder_hint: bool = True
    """Add a short system message telling the model to keep placeholders verbatim (only when placeholders were sent)."""
    """Let clients pick a policy with the X-PII-Policy header (otherwise only tenant/route decide)."""
    tenants_file: Path | None = None

    # ---- detection
    ner_enabled: bool = True
    ner_model: str = DEFAULT_NER_MODEL
    ner_threads: int = 4
    ner_preload: bool = True
    """Load the NER model at startup instead of on the first request."""
    llm_detector_provider: ProviderKind = "openrouter"
    llm_detector_model: str = DEFAULT_FREE_MODEL
    llm_detector_fallback_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_FREE_FALLBACKS)
    )

    # ---- vault
    vault_backend: Literal["memory", "redis", "file"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    vault_dir: Path = Path(".cache/vault")
    """Directory of the `file` backend (CLI redact -> restore across invocations)."""
    vault_key: str | None = None
    """URL-safe base64 AES-256 key. Without it a random key is generated per process (sessions die on restart)."""
    vault_ttl_seconds: int = 3600
    hash_key: str | None = None
    """Secret for the `hash` action and synthetic seeds; derived from the vault key if unset."""

    # ---- audit and dashboard
    audit_file: Path | None = None
    audit_max_entries: int = 2000
    results_dir: Path = Path("results")

    # ---- real-API budget (eval, LLM detector, playground)
    llm_cache_dir: Path = Path(".cache/llm")
    llm_ledger: Path = Path("results/calls.jsonl")
    llm_max_calls: int = 300
    llm_min_seconds_between_requests: float = 3.0
    llm_max_retries: int = 4

    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL", "PII_SHIELD_LOG_LEVEL"))
    log_format: Literal["console", "json"] = Field(
        default="console", validation_alias=AliasChoices("LOG_FORMAT", "PII_SHIELD_LOG_FORMAT")
    )

    _split_lists = field_validator("upstream_fallback_models", "llm_detector_fallback_models", mode="before")(_split)

    def upstream_default_model(self) -> str:
        if self.upstream_model:
            return self.upstream_model
        return DEFAULT_MODELS[self.upstream_provider]

    def has_key(self, provider: ProviderKind) -> bool:
        return provider == "fake" or bool(
            {"openrouter": self.openrouter_api_key, "openai": self.openai_api_key, "anthropic": self.anthropic_api_key}[
                provider
            ]
        )
