"""AES-256-GCM envelope for session mappings at rest.

The session id is bound as associated data, so a ciphertext copied under another session's key name
fails to decrypt instead of silently leaking the other conversation's values.
"""

from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_NONCE_BYTES = 12
_VERSION = b"\x01"


class VaultCryptoError(RuntimeError):
    pass


def generate_key() -> str:
    """A new random 256-bit key, URL-safe base64 (the format of PII_SHIELD_VAULT_KEY)."""
    return base64.urlsafe_b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")


def decode_key(value: str) -> bytes:
    try:
        raw = base64.urlsafe_b64decode(value.encode("ascii") + b"=" * (-len(value) % 4))
    except (ValueError, UnicodeEncodeError) as exc:
        raise VaultCryptoError("vault key must be URL-safe base64") from exc
    if len(raw) != 32:
        raise VaultCryptoError(f"vault key must decode to 32 bytes (AES-256), got {len(raw)}")
    return raw


class Cipher:
    def __init__(self, key: bytes) -> None:
        self._aead = AESGCM(key)

    def encrypt(self, plaintext: bytes, associated_data: str) -> bytes:
        nonce = os.urandom(_NONCE_BYTES)
        return _VERSION + nonce + self._aead.encrypt(nonce, plaintext, associated_data.encode("utf-8"))

    def decrypt(self, blob: bytes, associated_data: str) -> bytes:
        if not blob.startswith(_VERSION) or len(blob) < 1 + _NONCE_BYTES + 16:
            raise VaultCryptoError("unknown vault record format")
        nonce, ciphertext = blob[1 : 1 + _NONCE_BYTES], blob[1 + _NONCE_BYTES :]
        try:
            return self._aead.decrypt(nonce, ciphertext, associated_data.encode("utf-8"))
        except InvalidTag as exc:
            raise VaultCryptoError("vault record failed authentication (wrong key or tampered)") from exc
