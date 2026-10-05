"""Command line: `pii-shield redact | restore | serve | eval | gold | models | policies | keygen | download-model`."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

from pii_shield.config import DEFAULT_FREE_FALLBACKS, DEFAULT_FREE_MODEL, Settings
from pii_shield.logging_config import configure_logging

if TYPE_CHECKING:
    from pii_shield.eval.runner import EvalRunner

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="PII Shield: redact personal data before it reaches an LLM."
)
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation on the hand-labeled gold set.")
gold_app = typer.Typer(no_args_is_help=True, help="Build and inspect the gold set.")
models_app = typer.Typer(no_args_is_help=True, help="Free OpenRouter models.")
app.add_typer(eval_app, name="eval")
app.add_typer(gold_app, name="gold")
app.add_typer(models_app, name="models")
console = Console()
err = Console(stderr=True)

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Evaluation config YAML.")]


def _settings(**overrides: object) -> Settings:
    settings = Settings(**overrides)  # type: ignore[arg-type]
    configure_logging(settings.log_level, settings.log_format)
    return settings


def _read_text(text: str | None, file: Path | None) -> str:
    if file is not None:
        return file.read_text(encoding="utf-8")
    if text is None or text == "-":
        return sys.stdin.read()
    return text


@app.command()
def redact(
    text: Annotated[str | None, typer.Argument(help="Text to redact; '-' or omitted reads stdin.")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="Read the text from a file.")] = None,
    policy: Annotated[str | None, typer.Option("--policy", "-p")] = None,
    session: Annotated[
        str | None, typer.Option("--session", "-s", help="Session id (reuse to keep placeholders consistent).")
    ] = None,
    ner: Annotated[bool, typer.Option("--ner/--no-ner", help="Use the GLiNER model (needs the `ner` extra).")] = True,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full result as JSON (no original values).")
    ] = False,
) -> None:
    """Redact a text. The session is stored in the file vault so `restore` can reverse it later."""
    from pii_shield.gateway.runtime import build_pipeline, build_vault
    from pii_shield.policy import PolicySet
    from pii_shield.shield import Shield, derive_secret

    settings = _settings(vault_backend="file", ner_enabled=ner, ner_preload=False)
    vault, key, ephemeral = build_vault(settings)
    if ephemeral:
        err.print(
            "[yellow]PII_SHIELD_VAULT_KEY is not set: this session cannot be restored later "
            "(run `pii-shield keygen`).[/]"
        )
    pipeline, _ = build_pipeline(settings)
    shield = Shield(
        policies=PolicySet.from_dir(settings.policies_dir, settings.default_policy),
        pipeline=pipeline,
        vault=vault,
        secret=derive_secret(key, settings.hash_key),
    )
    result = shield.redact(_read_text(text, file), policy=policy, session_id=session)
    if as_json:
        payload = result.model_dump(mode="json")
        for entity in payload["entities"]:
            entity.pop("original", None)
        console.print_json(json.dumps(payload, ensure_ascii=False))
    else:
        console.print(result.text, markup=False, highlight=False)
        err.print(
            f"session {result.session_id} · policy {result.policy} · {len(result.entities)} entities {result.counts()}"
        )
    if result.blocked:
        err.print(f"[red]Blocked by policy: {', '.join(result.block_reasons)}[/]")
        raise typer.Exit(2)


@app.command()
def restore(
    text: Annotated[str | None, typer.Argument(help="Text with placeholders; '-' or omitted reads stdin.")] = None,
    session: Annotated[str, typer.Option("--session", "-s", help="Session id printed by `redact`.")] = "",
    file: Annotated[Path | None, typer.Option("--file", "-f")] = None,
) -> None:
    """Put the original values back (needs the same PII_SHIELD_VAULT_KEY as `redact`)."""
    from pii_shield.gateway.runtime import build_vault

    settings = _settings(vault_backend="file")
    if not settings.vault_key:
        err.print("[red]Set PII_SHIELD_VAULT_KEY (the key used by `redact`); generate one with `pii-shield keygen`.[/]")
        raise typer.Exit(1)
    if not session:
        err.print("[red]--session is required[/]")
        raise typer.Exit(1)
    vault, _, _ = build_vault(settings)
    from pii_shield.anonymize.restore import restore_text

    with vault.session(session) as state:
        if not state.entries:
            err.print(f"[yellow]Session {session} is empty or expired.[/]")
        restored, report = restore_text(_read_text(text, file), state)
    console.print(restored, markup=False, highlight=False)
    if report.unknown:
        err.print(f"[yellow]Unknown placeholders left as-is: {', '.join(report.unknown)}[/]")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes (development).")] = False,
) -> None:
    """Run the gateway, API and dashboard."""
    import uvicorn

    uvicorn.run("pii_shield.gateway.app:create_default_app", factory=True, host=host, port=port, reload=reload)


@app.command()
def keygen() -> None:
    """Print a new random AES-256 vault key (for PII_SHIELD_VAULT_KEY)."""
    from pii_shield.vault.crypto import generate_key

    console.print(generate_key())


@app.command("download-model")
def download_model(model: Annotated[str | None, typer.Argument()] = None) -> None:
    """Download the NER model (PyTorch weights only) into the Hugging Face cache."""
    from pii_shield.detect.ner import fetch_model

    settings = _settings()
    path = fetch_model(model or settings.ner_model)
    console.print(f"NER model ready: {path}")


@app.command()
def policies(name: Annotated[str | None, typer.Argument()] = None) -> None:
    """List policies, or show one."""
    from pii_shield.entities import EntityType
    from pii_shield.policy import PolicySet

    settings = _settings()
    policy_set = PolicySet.from_dir(settings.policies_dir, settings.default_policy)
    if name:
        policy = policy_set.get(name)
        table = Table(title=f"{policy.name}: {policy.description}")
        for column in ("Entity", "Action", "Threshold"):
            table.add_column(column)
        for kind in EntityType:
            rule = policy.rule(kind)
            table.add_row(
                kind.value,
                rule.action.value + (f" (keep last {rule.keep_last})" if rule.keep_last else ""),
                f"{policy.threshold(kind):.2f}",
            )
        console.print(table)
        return
    for policy in policy_set:
        marker = " (default)" if policy.name == policy_set.default_name else ""
        console.print(f"[bold]{policy.name}[/]{marker}: {policy.description}")


# ------------------------------------------------------------------------------------------- gold
@gold_app.command("build")
def gold_build(
    source: Annotated[Path, typer.Option()] = Path("data/gold/source"),
    output: Annotated[Path, typer.Option()] = Path("data/gold/gold.jsonl"),
) -> None:
    """Compile the hand-written markup into JSONL with character offsets."""
    from pii_shield.eval.gold import build_gold, gold_stats, write_gold

    docs = build_gold(source)
    write_gold(docs, output)
    console.print_json(json.dumps(gold_stats(docs)))


@gold_app.command("stats")
def gold_stats_command(path: Annotated[Path, typer.Argument()] = Path("data/gold/gold.jsonl")) -> None:
    from pii_shield.eval.gold import gold_stats, load_gold

    console.print_json(json.dumps(gold_stats(load_gold(path))))


# ------------------------------------------------------------------------------------------- eval
def _runner(config: Path) -> EvalRunner:
    from pii_shield.eval.config import load_eval_config
    from pii_shield.eval.runner import EvalRunner

    return EvalRunner(load_eval_config(config), _settings())


@eval_app.command("detection")
def eval_detection(
    config: ConfigOption = Path("configs/eval.yaml"),
    only: Annotated[
        str | None, typer.Option(help="Comma-separated detector configs to (re)run, e.g. patterns,patterns+ner.")
    ] = None,
) -> None:
    """P/R/F1 per entity type and detector configuration (the LLM config makes real API calls)."""
    from pii_shield.eval.report import write_report

    runner = _runner(config)
    report = runner.detection(only.split(",") if only else None)
    for c in report.configs:
        p = c.partial.overall
        console.print(
            f"{c.label:32} P={p.precision or 0:.3f} R={p.recall or 0:.3f} F1={p.f1 or 0:.3f}  "
            f"{c.latency_ms['mean']:.1f} ms/doc"
        )
    runner.calls_summary()
    console.print(f"report: {write_report(runner.out)}")


@eval_app.command("leak")
def eval_leak(config: ConfigOption = Path("configs/eval.yaml")) -> None:
    """Gold set through the gateway with a recording upstream: how much PII reaches the provider."""
    from pii_shield.eval.report import write_report

    runner = _runner(config)
    for r in runner.leak():
        console.print(
            f"{r.policy:24} {r.detectors:24} {r.channel:12} leaked {r.leaked}/{r.protected_values} = {r.leak_rate:.2%}"
        )
    console.print(f"report: {write_report(runner.out)}")


@eval_app.command("utility")
def eval_utility(config: ConfigOption = Path("configs/eval.yaml")) -> None:
    """Real-model answer quality with vs without redaction, judged by a second model (real API calls)."""
    from pii_shield.eval.report import write_report

    runner = _runner(config)
    report = runner.utility()
    console.print(
        f"verdicts {report.verdicts} · restore accuracy {report.restore_accuracy} · "
        f"name fidelity {report.name_fidelity}"
    )
    runner.calls_summary()
    console.print(f"report: {write_report(runner.out)}")


@eval_app.command("report")
def eval_report(directory: Annotated[Path, typer.Argument()] = Path("results")) -> None:
    """Re-render results/report.md from the JSON artifacts."""
    from pii_shield.eval.report import write_report

    console.print(f"report: {write_report(directory)}")


# ----------------------------------------------------------------------------------------- models
@models_app.command("free")
def models_free(
    smoke: Annotated[int, typer.Option(help="Smoke-test this many models with one tiny real call each.")] = 0,
    candidates: Annotated[
        str | None, typer.Option(help="Comma-separated ids to smoke-test (default: the configured ones).")
    ] = None,
    ledger: Annotated[Path, typer.Option(help="Call ledger for the smoke calls.")] = Path("results/calls.jsonl"),
) -> None:
    """List OpenRouter models whose id ends in ':free' (no key needed); optionally smoke-test a few."""
    response = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
    response.raise_for_status()
    free = [m for m in response.json()["data"] if str(m["id"]).endswith(":free")]
    table = Table(title=f"{len(free)} free models")
    for column in ("id", "context", "structured output"):
        table.add_column(column)
    for model in free:
        params = model.get("supported_parameters") or []
        table.add_row(
            model["id"],
            str(model.get("context_length")),
            "yes" if "response_format" in params or "structured_outputs" in params else "",
        )
    console.print(table)
    if smoke:
        ids = candidates.split(",") if candidates else [DEFAULT_FREE_MODEL, *DEFAULT_FREE_FALLBACKS]
        asyncio.run(_smoke(ids[:smoke], ledger))


async def _smoke(ids: list[str], ledger_path: Path) -> None:
    from pii_shield.providers.base import ProviderError, ensure_free_models
    from pii_shield.providers.factory import build_provider
    from pii_shield.providers.resilient import CallLedger, ResilientProvider, Throttle

    ensure_free_models(ids)
    settings = _settings()
    ledger = CallLedger(ledger_path, settings.llm_max_calls)
    throttle = Throttle(settings.llm_min_seconds_between_requests)
    results = []
    for model in ids:
        provider = ResilientProvider(
            build_provider(settings, kind="openrouter", model=model, fallback_models=[]),
            ledger=ledger,
            throttle=throttle,
            max_retries=1,
            tag="smoke",
        )
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "max_tokens": 200,
            "reasoning": {"effort": "low", "exclude": True},
        }
        try:
            answer = await provider.complete(body)
            text = (answer["choices"][0]["message"].get("content") or "").strip()[:40]
            results.append({"model": model, "ok": True, "served": answer.get("model"), "answer": text})
        except ProviderError as exc:
            results.append({"model": model, "ok": False, "error": str(exc)[:160]})
    for row in results:
        console.print(row)
    out = ledger_path.parent / "smoke.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


@app.command()
def calls(ledger: Annotated[Path, typer.Argument()] = Path("results/calls.jsonl")) -> None:
    """Summarize the real-API call ledger."""
    from pii_shield.eval.runner import summarize_ledger

    console.print_json(json.dumps(summarize_ledger(ledger)))


if __name__ == "__main__":
    app()
