"""The UserPromptSubmit hook's stdin/stdout JSON contract (https://code.claude.com/docs/en/hooks)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from pii_shield.claude_hook import run

CARD = "4111 1111 1111 1111"
EMAIL = "anna.petrova@gmail.com"


def event(prompt: object, **extra: Any) -> str:
    payload = {
        "session_id": "abc123",
        "transcript_path": "/tmp/transcript.jsonl",
        "cwd": "/tmp",
        "permission_mode": "default",
        "hook_event_name": "UserPromptSubmit",
        "prompt": prompt,
        **extra,
    }
    return json.dumps(payload)


def test_clean_prompt_passes_silently() -> None:
    result = run(event("Write a function to calculate the factorial of a number"))
    assert (result.exit_code, result.stdout, result.stderr) == (0, "", "")


def test_block_mode_blocks_without_echoing_values() -> None:
    result = run(event(f"Refund the card {CARD} please"), ["--mode", "block"])
    assert result.exit_code == 0 and result.stderr == ""
    out = json.loads(result.stdout)
    assert out["decision"] == "block" and "CREDIT_CARD x1" in out["reason"]
    assert out["hookSpecificOutput"] == {"hookEventName": "UserPromptSubmit", "suppressOriginalPrompt": True}
    assert CARD not in result.stdout and "1111" not in result.stdout


def test_default_mode_is_block_and_default_types_skip_contact_data() -> None:
    assert json.loads(run(event(f"key sk-proj-{'a1B2' * 8}")).stdout)["decision"] == "block"
    assert run(event(f"Mail {EMAIL} or call +44 7911 123456")).stdout == ""  # EMAIL/PHONE not selected


def test_selected_types_only() -> None:
    result = run(event(f"Mail {EMAIL}, card {CARD}"), ["--types", "EMAIL"])
    reason = json.loads(result.stdout)["reason"]
    assert "EMAIL x1" in reason and "CREDIT_CARD" not in reason and EMAIL not in result.stdout
    assert json.loads(run(event(f"Mail {EMAIL}"), ["--types", "all"]).stdout)["decision"] == "block"


def test_warn_mode_lets_the_prompt_through_with_context() -> None:
    result = run(event(f"Mail {EMAIL}, card {CARD}"), ["--mode=warn", "--types", "EMAIL,CREDIT_CARD"])
    assert result.exit_code == 0
    out = json.loads(result.stdout)
    assert "decision" not in out and "CREDIT_CARD x1, EMAIL x1" in out["systemMessage"]
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "Do not repeat these values" in out["hookSpecificOutput"]["additionalContext"]
    assert EMAIL not in result.stdout and CARD not in result.stdout


def test_policy_allow_and_deny_lists_apply() -> None:
    # support-chat allow-lists the operator's own addresses and deny-lists the internal code name.
    assert run(event("Forward it to help@brightloop.com"), ["--types", "EMAIL"]).stdout == ""
    blocked = json.loads(run(event("Status of Project Falcon?"), ["--types", "CUSTOM"]).stdout)
    assert "CUSTOM x1" in blocked["reason"]


def test_custom_policy_file(tmp_path: Path) -> None:
    policy = tmp_path / "team.yaml"
    policy.write_text("name: team\nallow_list:\n  - {value: 'ops@acme.io', entity: EMAIL}\n", encoding="utf-8")
    assert run(event("ping ops@acme.io"), ["--types", "EMAIL", "--policy", str(policy)]).stdout == ""
    assert run(event(f"ping {EMAIL}"), ["--types", "EMAIL", "--policy", str(policy)]).stdout != ""


def test_placeholders_are_not_pii() -> None:
    assert run(event("Reply to <EMAIL_1> about <CREDIT_CARD_1>"), ["--types", "all"]).stdout == ""


def test_other_events_are_ignored() -> None:
    assert run(event(f"card {CARD}", hook_event_name="SessionStart")).stdout == ""


@pytest.mark.parametrize(
    ("stdin", "argv"),
    [
        ("not json", []),
        (json.dumps(["a list"]), []),
        (event(None), []),
        (event("hi"), ["--types", "PASSPORT"]),
        (event("hi"), ["--policy", "nope"]),
        (event("hi"), ["--bogus"]),
    ],
)
def test_failures_fail_closed_in_block_mode(stdin: str, argv: list[str]) -> None:
    result = run(stdin, argv)
    out = json.loads(result.stdout)
    assert result.exit_code == 0 and out["decision"] == "block" and "hook error" in out["reason"]
    assert out["hookSpecificOutput"]["suppressOriginalPrompt"] is True


def test_failures_are_non_blocking_errors_in_warn_mode() -> None:
    result = run("not json", ["--mode", "warn"])
    assert result.exit_code == 1 and result.stdout == "" and "JSONDecodeError" in result.stderr
    assert run(event("hi"), ["--mode", "warn", "--types", "PASSPORT"]).exit_code == 1


def test_console_script_contract() -> None:
    """The real process: stdout is exactly one JSON object, exit code 0."""
    proc = subprocess.run(
        [sys.executable, "-m", "pii_shield.claude_hook", "--types", "CREDIT_CARD"],
        input=event(f"card {CARD}"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["decision"] == "block" and proc.stdout.count("\n") == 1
