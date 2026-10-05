"""NER for names, street addresses and organizations with a GLiNER PII model on CPU.

GLiNER is a span-extraction model that takes the label names as a prompt, so the label wording is
configurable. The model and torch are optional (`pip install "pii-shield[ner]"`); without them the
shield runs patterns only and says so in /health and on the dashboard.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pii_shield.entities import EntityType, Span
from pii_shield.logging_config import get_logger

log = get_logger(__name__)

DEFAULT_NER_MODEL = "knowledgator/gliner-pii-base-v1.0"
DEFAULT_LABELS: dict[str, EntityType] = {
    "person": EntityType.PERSON,
    "street address": EntityType.ADDRESS,
    "organization": EntityType.ORGANIZATION,
}
_MAX_CHUNK_CHARS = 900
_SENTENCE_END = re.compile(r"(?<=[.!?\n])\s+")


# Only the PyTorch checkpoint, config and tokenizer are needed; some repos also carry 1+ GB of ONNX exports.
_DOWNLOAD_IGNORE = ["onnx/*", "*.onnx", "*.md", "trainer_state.json", "*.h5", "*.msgpack"]


class NerUnavailableError(RuntimeError):
    """The `ner` extra is not installed or the model could not be loaded."""


def fetch_model(model_name: str, *, local_files_only: bool = False) -> str:
    """Local directory of the model: the path itself, or a Hugging Face snapshot without ONNX files."""
    if Path(model_name).is_dir():
        return model_name
    from huggingface_hub import snapshot_download

    return str(snapshot_download(model_name, ignore_patterns=_DOWNLOAD_IGNORE, local_files_only=local_files_only))


def model_cached(model_name: str) -> bool:
    """True if the model can be loaded without network access (used to skip NER tests in CI)."""
    if not ner_installed():
        return False
    try:
        fetch_model(model_name, local_files_only=True)
    except Exception:  # missing files raise several different hub errors
        return False
    return True


def ner_installed() -> bool:
    try:
        import gliner  # noqa: F401
    except ImportError:
        return False
    return True


def chunk_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> Iterator[tuple[int, str]]:
    """Split long text on sentence boundaries into (offset, chunk) pieces the model can see whole."""
    if len(text) <= max_chars:
        yield 0, text
        return
    start = 0
    current_start = 0
    for match in _SENTENCE_END.finditer(text):
        if match.end() - current_start > max_chars and start > current_start:
            yield current_start, text[current_start:start]
            current_start = start
        start = match.end()
    while len(text) - current_start > max_chars:
        # A single very long "sentence" (logs, CSV): cut at the last whitespace before the limit.
        cut = text.rfind(" ", current_start, current_start + max_chars)
        cut = cut if cut > current_start else current_start + max_chars
        yield current_start, text[current_start:cut]
        current_start = cut
    if current_start < len(text):
        yield current_start, text[current_start:]


@dataclass
class GlinerDetector:
    """Wraps a GLiNER model. Thread-safe lazy loading; inference itself is serialized per process."""

    model_name: str = DEFAULT_NER_MODEL
    labels: Mapping[str, EntityType] = field(default_factory=lambda: dict(DEFAULT_LABELS))
    min_score: float = 0.3
    """Floor passed to the model; per-type policy thresholds filter further."""
    threads: int = 4
    """torch intra-op threads (process-wide). More is not faster for short texts and hurts under load."""
    name: str = "ner"
    _model: Any = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    load_seconds: float | None = field(default=None, init=False)

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from gliner import GLiNER
            except ImportError as exc:
                raise NerUnavailableError('GLiNER is not installed: pip install "pii-shield[ner]"') from exc
            started = time.monotonic()
            if self.threads > 0:
                import torch

                torch.set_num_threads(self.threads)
            try:
                model = GLiNER.from_pretrained(fetch_model(self.model_name))
            except Exception as exc:  # network, missing files, incompatible checkpoint
                raise NerUnavailableError(f"could not load NER model {self.model_name!r}: {exc}") from exc
            model.eval()
            self._model = model
            self.load_seconds = round(time.monotonic() - started, 2)
            log.info("ner.loaded", model=self.model_name, seconds=self.load_seconds)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def detect(self, text: str) -> list[Span]:
        if not text.strip():
            return []
        self.load()
        spans: list[Span] = []
        label_names = list(self.labels)
        with self._lock:
            for offset, chunk in chunk_text(text):
                if not chunk.strip():
                    continue
                entities = self._model.predict_entities(chunk, label_names, threshold=self.min_score)
                for entity in entities:
                    kind = self.labels.get(str(entity["label"]))
                    if kind is None:
                        continue
                    start, end = offset + int(entity["start"]), offset + int(entity["end"])
                    spans.append(
                        Span(
                            start=start,
                            end=end,
                            type=kind,
                            text=text[start:end],
                            score=round(min(max(float(entity["score"]), 0.0), 1.0), 4),
                            source=self.name,
                        )
                    )
        return spans
