import time

import fakeredis
import pytest

from pii_shield.entities import EntityType
from pii_shield.shield import Shield
from pii_shield.vault.crypto import Cipher, VaultCryptoError, decode_key, generate_key
from pii_shield.vault.session import SessionState
from pii_shield.vault.store import FileBackend, MemoryBackend, RedisBackend, Vault

KEY = b"k" * 32


def test_placeholders_consistent_across_turns(shield: Shield) -> None:
    with shield.session("conv-1") as s:
        first = s.redact("Hi, I'm Anna Petrova, anna.petrova@gmail.com.")
    with shield.session("conv-1") as s:
        second = s.redact("Again Anna Petrova here; mail anna.petrova@gmail.com, or call +44 7911 123456.")
        answer = s.restore("Dear <PERSON_1>, we will reply to <EMAIL_1> and call <PHONE_1>.")
    assert first.text == "Hi, I'm <PERSON_1>, <EMAIL_1>."
    assert second.text == "Again <PERSON_1> here; mail <EMAIL_1>, or call <PHONE_1>."
    assert answer == "Dear Anna Petrova, we will reply to anna.petrova@gmail.com and call +44 7911 123456."


def test_name_variants_link_to_one_placeholder(shield: Shield) -> None:
    with shield.session("conv-2") as s:
        result = s.redact("Anna Petrova wrote in. Ms. Petrova asked for a refund; Anna is waiting.")
        assert result.text == "<PERSON_1> wrote in. Ms. <PERSON_1> asked for a refund; <PERSON_1> is waiting."
        assert s.restore("Thanks <PERSON_1>") == "Thanks Anna Petrova"


def test_ambiguous_first_name_is_not_linked() -> None:
    state = SessionState(session_id="s")
    for name in ("Anna Petrova", "Anna Schmidt"):
        state.assign_placeholder(state.get_or_create(EntityType.PERSON, name))
    entry = state.get_or_create(EntityType.PERSON, "Anna")
    assert state.assign_placeholder(entry) == "<PERSON_3>"


def test_vault_encrypts_at_rest_and_binds_session_id() -> None:
    backend = MemoryBackend()
    vault = Vault(backend, KEY)
    with vault.session("s1") as state:
        state.assign_placeholder(state.get_or_create(EntityType.EMAIL, "secret.person@example.com"))
    blob = backend.get("s1")
    assert blob is not None and b"secret.person" not in blob
    backend.set("s2", blob, 60)  # copying a record under another session id must not decrypt
    with pytest.raises(VaultCryptoError):
        vault.load("s2")
    with pytest.raises(VaultCryptoError):
        Vault(backend, b"x" * 32).load("s1")


def test_vault_ttl_expiry_in_memory() -> None:
    now = [1000.0]
    backend = MemoryBackend(clock=lambda: now[0])
    vault = Vault(backend, KEY, ttl_seconds=60)
    with vault.session("s1") as state:
        state.assign_placeholder(state.get_or_create(EntityType.PERSON, "Anna"))
    now[0] += 59
    assert vault.load("s1").entries
    now[0] += 2
    assert vault.load("s1").entries == []
    assert backend.count() == 0


def test_file_backend_roundtrip_and_ttl(tmp_path) -> None:  # type: ignore[no-untyped-def]
    now = [time.time()]
    backend = FileBackend(tmp_path, clock=lambda: now[0])
    vault = Vault(backend, KEY, ttl_seconds=10)
    with vault.session("cli") as state:
        state.assign_placeholder(state.get_or_create(EntityType.EMAIL, "a@b.co"))
    assert not any(b"a@b.co" in p.read_bytes() for p in tmp_path.iterdir())
    assert vault.load("cli").by_placeholder("<EMAIL_1>") is not None
    now[0] += 11
    assert vault.load("cli").entries == []


def test_redis_backend_with_fakeredis() -> None:
    backend = RedisBackend.from_client(fakeredis.FakeRedis())
    vault = Vault(backend, KEY, ttl_seconds=30)
    with vault.session("r1") as state:
        state.assign_placeholder(state.get_or_create(EntityType.PHONE, "+44 7911 123456"))
    assert vault.load("r1").by_placeholder("<PHONE_1>") is not None
    assert backend.count() == 1
    vault.delete("r1")
    assert vault.load("r1").entries == []


def test_session_lock_serializes_requests() -> None:
    backend = MemoryBackend()
    with backend.lock("s", 1.0), pytest.raises(TimeoutError), backend.lock("s", 0.05):
        pass


def test_key_helpers() -> None:
    key = generate_key()
    assert len(decode_key(key)) == 32
    with pytest.raises(VaultCryptoError):
        decode_key("dG9vLXNob3J0")
    cipher = Cipher(KEY)
    assert cipher.decrypt(cipher.encrypt(b"x", "a"), "a") == b"x"
