"""Audit log without PII: what was found and done per request, never the values.

Each record holds entity counts by type and action, the policy, latencies, upstream model, and a keyed
hash of the request body (to correlate a complaint with a request without storing it). The session id
is hashed too, so the log cannot be used to look up a conversation's vault entry.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import uuid
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from pii_shield.logging_config import get_logger
from pii_shield.shield import RedactionResult

log = get_logger(__name__)


class AuditRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))
    route: str
    tenant: str | None = None
    policy: str
    session_hash: str
    request_hash: str
    entity_counts: dict[str, int] = Field(default_factory=dict)
    action_counts: dict[str, int] = Field(default_factory=dict)
    detectors: list[str] = Field(default_factory=list)
    detector_errors: list[str] = Field(default_factory=list)
    """Names of failed detectors only (error messages can quote input)."""
    blocked: bool = False
    block_reasons: list[str] = Field(default_factory=list)
    upstream: str | None = None
    upstream_model: str | None = None
    stream: bool = False
    status: int = 200
    restored: int = 0
    unknown_placeholders: int = 0
    redact_ms: float = 0.0
    upstream_ms: float | None = None
    total_ms: float = 0.0


class AuditLog:
    def __init__(self, key: bytes, max_entries: int = 2000, path: Path | None = None) -> None:
        self._key = key
        self._records: deque[AuditRecord] = deque(maxlen=max_entries)
        self._path = path
        self._lock = threading.Lock()

    def digest(self, data: bytes | str) -> str:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        return hmac.new(self._key, raw, hashlib.sha256).hexdigest()[:24]

    def new_record(
        self, *, route: str, result: RedactionResult, request_body: bytes | str, tenant: str | None
    ) -> AuditRecord:
        return AuditRecord(
            route=route,
            tenant=tenant,
            policy=result.policy,
            session_hash=self.digest("session:" + result.session_id),
            request_hash=self.digest(request_body),
            entity_counts=result.counts(),
            action_counts=result.action_counts(),
            detectors=[name for name in result.timings_ms if name != "total"],
            detector_errors=sorted(result.detector_errors),
            blocked=result.blocked,
            block_reasons=list(result.block_reasons),
            redact_ms=result.timings_ms.get("total", 0.0),
        )

    def add(self, record: AuditRecord) -> None:
        with self._lock:
            self._records.append(record)
            if self._path is None:
                return
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(record.model_dump_json() + "\n")
            except OSError as exc:  # a full or read-only disk must not fail the user's request
                log.error("audit.write_failed", path=str(self._path), error=type(exc).__name__)

    def preload(self, records: list[AuditRecord]) -> None:
        """Show records persisted by an earlier process (not written again)."""
        with self._lock:
            self._records.extend(records)

    def recent(self, limit: int = 100) -> list[AuditRecord]:
        with self._lock:
            return list(self._records)[-limit:][::-1]

    def summary(self) -> dict[str, object]:
        with self._lock:
            records = list(self._records)
        entities: Counter[str] = Counter()
        for record in records:
            entities.update(record.entity_counts)
        return {
            "requests": len(records),
            "blocked": sum(r.blocked for r in records),
            "entities": dict(entities.most_common()),
            "policies": dict(Counter(r.policy for r in records)),
        }

    @staticmethod
    def load_jsonl(path: Path) -> list[AuditRecord]:
        if not path.exists():
            return []
        return [AuditRecord.model_validate(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
