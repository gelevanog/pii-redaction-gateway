"""Evaluation config (configs/eval.yaml): gold set, detector configurations, leak and utility settings."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pii_shield.config import DEFAULT_FREE_FALLBACKS, DEFAULT_FREE_MODEL, ProviderKind
from pii_shield.detect.ner import DEFAULT_NER_MODEL
from pii_shield.providers.base import FreeModelGuardError, ensure_free_models


class DetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    label: str
    patterns: bool = True
    ner: bool = False
    llm: bool = False


class LlmModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderKind = "openrouter"
    model: str = DEFAULT_FREE_MODEL
    fallback_models: list[str] = Field(default_factory=lambda: list(DEFAULT_FREE_FALLBACKS))

    @property
    def all_models(self) -> list[str]:
        return [self.model, *self.fallback_models]


class UtilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = "reply"
    documents: int = 20
    domains: list[str] = Field(default_factory=lambda: ["support", "email", "chat"])
    model: LlmModelConfig = Field(default_factory=LlmModelConfig)
    judge: LlmModelConfig = Field(
        default_factory=lambda: LlmModelConfig(model="google/gemma-4-31b-it:free", fallback_models=[])
    )
    policy: str = "support-chat"
    max_tokens: int = 3000
    noise_floor: bool = True
    """Also judge two samples of the original-text answer against each other (doubles baseline + judge calls)."""


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gold: Path = Path("data/gold/gold.jsonl")
    output_dir: Path = Path("results")
    policy: str = "support-chat"
    """Thresholds and allow-list used for the detection metrics."""
    ner_model: str = DEFAULT_NER_MODEL
    ner_compare: list[str] = Field(default_factory=list)
    """Other NER models to score (patterns + that model) for the model-choice table."""
    detectors: list[DetectorConfig] = Field(
        default_factory=lambda: [
            DetectorConfig(name="patterns", label="Patterns + validators"),
            DetectorConfig(name="patterns+ner", label="Patterns + NER", ner=True),
        ]
    )
    llm_detector: LlmModelConfig = Field(default_factory=LlmModelConfig)
    leak_policies: list[str] = Field(default_factory=lambda: ["support-chat", "strict-finance"])
    leak_detectors: list[str] = Field(default_factory=lambda: ["patterns", "patterns+ner"])
    utility: UtilityConfig = Field(default_factory=UtilityConfig)
    require_free_models: bool = True
    max_calls: int = 300
    min_seconds_between_requests: float = 3.0
    llm_concurrency: int = 3
    llm_subset_every: int = 2
    """The LLM-detector comparison runs on every n-th gold document (by id) to fit the free-tier call budget;
    all configurations are scored on that same subset."""

    @model_validator(mode="after")
    def _free_only(self) -> EvalConfig:
        if self.require_free_models:
            for llm in (self.llm_detector, self.utility.model, self.utility.judge):
                if llm.provider == "openrouter":
                    try:
                        ensure_free_models(llm.all_models)
                    except FreeModelGuardError as exc:
                        raise ValueError(str(exc)) from exc
                elif llm.provider in {"openai", "anthropic"}:
                    raise ValueError(f"require_free_models is on, but provider {llm.provider!r} is always paid")
        return self

    def detector(self, name: str) -> DetectorConfig:
        for config in self.detectors:
            if config.name == name:
                return config
        raise KeyError(f"unknown detector config {name!r}")


def load_eval_config(path: Path | str) -> EvalConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return EvalConfig.model_validate(raw)
