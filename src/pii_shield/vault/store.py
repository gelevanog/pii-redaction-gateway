"""Session vault backends: placeholder <-> original mappings per conversation, encrypted, with a TTL.

A backend stores opaque encrypted blobs; `Vault` does the (de)serialization and encryption, so the
in-memory and Redis backends never see a plaintext value.
"""

from __future__ import annotations

import hashlib
import struct
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from pii_shield.vault.crypto import Cipher
from pii_shield.vault.session import SessionState


class VaultBackend(Protocol):
    name: str

    def get(self, key: str) -> bytes | None: ...

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None: ...

    def delete(self, key: str) -> None: ...

    @contextmanager
    def lock(self, key: str, timeout_seconds: float) -> Iterator[None]: ...

    def count(self) -> int: ...


class MemoryBackend:
    """Process-local backend for a single gateway instance, tests and the library."""

    name = "memory"

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._data: dict[str, tuple[float, bytes]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._clock = clock or time.monotonic

    def _now(self) -> float:
        return self._clock()

    def get(self, key: str) -> bytes | None:
        with self._guard:
            item = self._data.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= self._now():
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        with self._guard:
            self._data[key] = (self._now() + ttl_seconds, value)
            if len(self._data) % 256 == 0:
                self._purge_locked()

    def delete(self, key: str) -> None:
        with self._guard:
            self._data.pop(key, None)

    def _purge_locked(self) -> None:
        now = self._now()
        for key in [k for k, (expires_at, _) in self._data.items() if expires_at <= now]:
            del self._data[key]

    def count(self) -> int:
        with self._guard:
            self._purge_locked()
            return len(self._data)

    @contextmanager
    def lock(self, key: str, timeout_seconds: float) -> Iterator[None]:
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        if not lock.acquire(timeout=timeout_seconds):
            raise TimeoutError(f"session is busy (lock wait > {timeout_seconds}s)")
        try:
            yield
        finally:
            lock.release()


class FileBackend:
    """Encrypted blobs in a local directory (CLI and single-host use). File names are hashed session ids."""

    name = "file"

    def __init__(self, directory: Path, clock: Callable[[], float] | None = None) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._clock = clock or time.time
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.directory / (hashlib.sha256(key.encode("utf-8")).hexdigest()[:32] + ".bin")

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        if not path.exists():
            return None
        raw = path.read_bytes()
        expires_at = struct.unpack(">d", raw[:8])[0]
        if expires_at <= self._clock():
            path.unlink(missing_ok=True)
            return None
        return raw[8:]

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(struct.pack(">d", self._clock() + ttl_seconds) + value)
        tmp.replace(path)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def count(self) -> int:
        return sum(1 for path in self.directory.glob("*.bin") if self._alive(path))

    def _alive(self, path: Path) -> bool:
        raw = path.read_bytes()[:8]
        return len(raw) == 8 and struct.unpack(">d", raw)[0] > self._clock()

    @contextmanager
    def lock(self, key: str, timeout_seconds: float) -> Iterator[None]:
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        if not lock.acquire(timeout=timeout_seconds):
            raise TimeoutError(f"session is busy (lock wait > {timeout_seconds}s)")
        try:
            yield
        finally:
            lock.release()


class RedisBackend:
    """Shared backend for several gateway replicas. Keys expire in Redis itself (SET ... EX)."""

    name = "redis"

    def __init__(self, url: str, prefix: str = "pii-shield:session:") -> None:
        import redis

        self._client = redis.Redis.from_url(url)
        self._prefix = prefix

    @classmethod
    def from_client(cls, client: object, prefix: str = "pii-shield:session:") -> RedisBackend:
        backend = cls.__new__(cls)
        backend._client = client  # type: ignore[assignment]
        backend._prefix = prefix
        return backend

    def get(self, key: str) -> bytes | None:
        value = self._client.get(self._prefix + key)
        return value if isinstance(value, bytes) else None

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        self._client.set(self._prefix + key, value, ex=ttl_seconds)

    def delete(self, key: str) -> None:
        self._client.delete(self._prefix + key)

    def count(self) -> int:
        return sum(1 for _ in self._client.scan_iter(match=self._prefix + "*", count=500))

    @contextmanager
    def lock(self, key: str, timeout_seconds: float) -> Iterator[None]:
        lock = self._client.lock(self._prefix + "lock:" + key, timeout=30, blocking_timeout=timeout_seconds)
        if not lock.acquire():
            raise TimeoutError(f"session is busy (lock wait > {timeout_seconds}s)")
        try:
            yield
        finally:
            lock.release()

    def ping(self) -> bool:
        return bool(self._client.ping())


class Vault:
    """Loads and saves `SessionState`s through a backend, encrypted with AES-256-GCM."""

    def __init__(self, backend: VaultBackend, key: bytes, ttl_seconds: int = 3600) -> None:
        self.backend = backend
        self.ttl_seconds = ttl_seconds
        self._cipher = Cipher(key)

    def load(self, session_id: str) -> SessionState:
        blob = self.backend.get(session_id)
        if blob is None:
            return SessionState(session_id=session_id)
        plaintext = self._cipher.decrypt(blob, associated_data=session_id)
        return SessionState.model_validate_json(plaintext)

    def save(self, state: SessionState) -> None:
        blob = self._cipher.encrypt(state.model_dump_json().encode("utf-8"), associated_data=state.session_id)
        self.backend.set(state.session_id, blob, self.ttl_seconds)

    def delete(self, session_id: str) -> None:
        self.backend.delete(session_id)

    @contextmanager
    def session(self, session_id: str, timeout_seconds: float = 10.0) -> Iterator[SessionState]:
        """Lock, load, yield for mutation, save. Requests of one conversation are serialized."""
        with self.backend.lock(session_id, timeout_seconds):
            state = self.load(session_id)
            yield state
            if state.dirty or state.entries:
                self.save(state)  # also refreshes the TTL of an active conversation
                state.dirty = False
