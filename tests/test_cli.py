from pathlib import Path

import pytest
from typer.testing import CliRunner

from pii_shield.cli import app
from pii_shield.vault.crypto import generate_key

runner = CliRunner()


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PII_SHIELD_VAULT_KEY", generate_key())
    monkeypatch.setenv("PII_SHIELD_VAULT_DIR", str(tmp_path / "vault"))
    return tmp_path


def test_redact_then_restore_across_invocations(cli_env: Path) -> None:
    redacted = runner.invoke(
        app, ["redact", "--no-ner", "--session", "cli-1", "Mail anna@gmail.com or call +44 7911 123456"]
    )
    assert redacted.exit_code == 0, redacted.output
    assert "Mail <EMAIL_1> or call <PHONE_1>" in redacted.stdout
    restored = runner.invoke(app, ["restore", "--session", "cli-1", "We wrote to <EMAIL_1>."])
    assert restored.exit_code == 0 and "We wrote to anna@gmail.com." in restored.stdout


def test_redact_json_has_no_original_values(cli_env: Path) -> None:
    result = runner.invoke(app, ["redact", "--no-ner", "--json", "card 4111 1111 1111 1111"])
    assert result.exit_code == 0 and "**** **** **** 1111" in result.stdout and '"original"' not in result.stdout


def test_blocked_redaction_exits_nonzero(cli_env: Path) -> None:
    result = runner.invoke(app, ["redact", "--no-ner", "--policy", "strict-finance", "card 4111 1111 1111 1111"])
    assert result.exit_code == 2


def test_policies_keygen_gold_and_report(cli_env: Path) -> None:
    assert "strict-finance" in runner.invoke(app, ["policies"]).stdout
    assert "CREDIT_CARD" in runner.invoke(app, ["policies", "strict-finance"]).stdout
    assert len(runner.invoke(app, ["keygen"]).stdout.strip()) == 44
    out = cli_env / "gold.jsonl"
    built = runner.invoke(app, ["gold", "build", "--output", str(out)])
    assert built.exit_code == 0 and out.exists() and '"documents"' in built.stdout
    assert runner.invoke(app, ["eval", "report", str(cli_env)]).exit_code == 0


def test_restore_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PII_SHIELD_VAULT_KEY", raising=False)
    assert runner.invoke(app, ["restore", "--session", "x", "hi"]).exit_code == 1
