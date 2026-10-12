"""Claude Code `UserPromptSubmit` hook: stop (or flag) prompts that contain personal data, before Claude sees them.

Claude Code runs the hook with the event as JSON on stdin and reads one JSON object from stdout
(https://code.claude.com/docs/en/hooks). Detection is the deterministic pattern layer only (no NER, no LLM),
so the hook answers in well under a second and well inside the 30-second hook timeout.

    pii-shield-prompt-hook --mode block --types CREDIT_CARD,IBAN,US_SSN,NATIONAL_ID,SECRET

- `block`: `{"decision": "block", ...}`; the prompt never reaches Claude. `suppressOriginalPrompt` keeps the
  text out of the block message (Claude Code still keeps it in the local transcript and prompt history).
- `warn`: the prompt goes through; the user sees a `systemMessage`, and Claude gets `additionalContext`
  asking it not to repeat or store the values.

Neither output contains a detected value, only types and counts. If the hook itself fails, `block` mode
fails closed (the prompt is blocked with an explanation) and `warn` mode exits 1 (a non-blocking error).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from pii_shield.detect.pipeline import DetectionPipeline
from pii_shield.entities import EntityType
from pii_shield.policy import DetectorToggles, Policy, PolicyError, load_policy
from pii_shield.shield import PACKAGED_POLICIES

DEFAULT_TYPES: tuple[EntityType, ...] = (
    EntityType.CREDIT_CARD,
    EntityType.IBAN,
    EntityType.US_SSN,
    EntityType.NATIONAL_ID,
    EntityType.SECRET,
)
EVENT = "UserPromptSubmit"


@dataclass
class HookResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class _ArgumentError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _ArgumentError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="pii-shield-prompt-hook",
        description="Claude Code UserPromptSubmit hook: block or warn on prompts containing personal data.",
    )
    parser.add_argument("--mode", choices=("block", "warn"), default="block")
    parser.add_argument(
        "--types",
        default=",".join(t.value for t in DEFAULT_TYPES),
        help="Comma-separated entity types to act on, or 'all'. Default: %(default)s.",
    )
    parser.add_argument(
        "--policy",
        default="support-chat",
        help="Packaged policy name or YAML path: thresholds, allow-list and deny-list (deny-list hits are CUSTOM).",
    )
    return parser


def _parse_types(raw: str) -> frozenset[EntityType]:
    if raw.strip().lower() == "all":
        return frozenset(EntityType)
    names = [item.strip().upper() for item in raw.split(",") if item.strip()]
    unknown = [n for n in names if n not in EntityType.__members__]
    if unknown or not names:
        valid = ", ".join(t.value for t in EntityType)
        raise _ArgumentError(f"unknown entity type(s) {', '.join(unknown) or '(none given)'}; valid: {valid}, all")
    return frozenset(EntityType(n) for n in names)


def _load_policy(name: str) -> Policy:
    path = Path(name) if name.endswith((".yaml", ".yml")) else PACKAGED_POLICIES / f"{name}.yaml"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in PACKAGED_POLICIES.glob("*.yaml")))
        raise PolicyError(f"policy {name!r} not found (packaged: {available})")
    policy = load_policy(path)
    # Patterns and the deny-list only: deterministic, fast, no model download, no API call.
    return policy.model_copy(update={"detectors": DetectorToggles(patterns=True, ner=False, llm=False)})


def _requested_mode(argv: list[str]) -> str:
    """The mode asked for, even when the arguments do not parse (so a typo still fails closed in block mode)."""
    if "--mode=warn" in argv or any(pair == ("--mode", "warn") for pair in pairwise(argv)):
        return "warn"
    return "block"


def scan(prompt: str, policy: Policy, types: frozenset[EntityType]) -> Counter[str]:
    outcome = DetectionPipeline().detect(prompt, policy)
    return Counter(span.type.value for span in outcome.spans if span.type in types)


def _summary(counts: Counter[str]) -> str:
    return ", ".join(f"{kind} x{n}" for kind, n in sorted(counts.items()))


def _json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _block(reason: str) -> HookResult:
    payload = {
        "decision": "block",
        "reason": reason,
        "hookSpecificOutput": {"hookEventName": EVENT, "suppressOriginalPrompt": True},
    }
    return HookResult(0, _json(payload))


def run(stdin: str, argv: list[str] | None = None) -> HookResult:
    """The whole hook as a pure function: event JSON in, exit code + stdout/stderr out."""
    argv = list(argv or [])
    mode = _requested_mode(argv)
    try:
        args = _parser().parse_args(argv)
        mode = args.mode
        types = _parse_types(args.types)
        policy = _load_policy(args.policy)
        event = json.loads(stdin)
        if not isinstance(event, dict):
            raise ValueError("hook input is not a JSON object")
        if event.get("hook_event_name", EVENT) != EVENT:
            return HookResult(0)  # registered under another event by mistake: stay out of the way
        prompt = event.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("hook input has no string `prompt` field")
        counts = scan(prompt, policy, types)
    except Exception as exc:  # any failure: block mode fails closed, warn mode reports a non-blocking error
        message = f"PII Shield hook error ({type(exc).__name__}): {str(exc)[:300]}"
        if mode == "block":
            return _block(f"{message}. The prompt was blocked because the hook runs in fail-closed block mode.")
        return HookResult(1, stderr=message)

    if not counts:
        return HookResult(0)
    found = _summary(counts)
    if mode == "block":
        return _block(
            f"PII Shield blocked this prompt before it reached Claude: it contains {found}. "
            "Remove or replace these values and send it again (for example, paste the output of "
            "`pii-shield redact --no-ner` instead). The values themselves are not shown here."
        )
    payload = {
        "systemMessage": f"PII Shield: this prompt contains {found}. It was sent to Claude anyway (warn mode).",
        "hookSpecificOutput": {
            "hookEventName": EVENT,
            "additionalContext": (
                f"PII Shield notice: the user's prompt contains personal data ({found}). Do not repeat these "
                "values, and do not write them to files, commits, logs or other tools unless the user explicitly "
                "asks for it."
            ),
        },
    }
    return HookResult(0, _json(payload))


def main() -> None:
    result = run(sys.stdin.read(), sys.argv[1:])
    if result.stdout:
        sys.stdout.write(result.stdout + "\n")
    if result.stderr:
        sys.stderr.write(result.stderr + "\n")
    sys.exit(result.exit_code)


if __name__ == "__main__":
    main()
