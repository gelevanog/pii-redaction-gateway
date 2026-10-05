"""Render results/*.json as Markdown tables (results/report.md, also used for the README)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _pct(value: Any, digits: int = 1) -> str:
    return "–" if value is None else f"{value * 100:.{digits}f}%"


def _f(value: Any) -> str:
    return "–" if value is None else f"{value:.2f}"


def _row(*cells: object) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _table(header: list[str], align: str) -> list[str]:
    return [_row(*header), "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]


def _ordinal(n: int) -> str:
    return {1: "", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _latency(config: dict[str, Any]) -> str:
    return f"{config['latency_ms']['mean']:.0f} / {config['latency_ms']['p95']:.0f} ms"


def _negatives(config: dict[str, Any]) -> str:
    return f"{config['negatives_with_findings']}/{config['negatives']}"


def detection_tables(detection: dict[str, Any]) -> list[str]:
    gold = detection["gold"]
    lines = [
        "## Detection (gold set)",
        "",
        f"{gold['documents']} documents, {gold['entities']} entities, {gold['negatives']} documents without PII. "
        f"Thresholds and allow-list: policy `{detection['policy']}`.",
        "",
    ]
    header = ["Configuration", "Precision", "Recall", "F1 strict", "F1 partial", "Any-type recall"]
    lines += _table([*header, "Clean docs flagged", "Latency"], "lrrrrrrr")
    for c in detection["configs"]:
        s, p = c["strict"]["overall"], c["partial"]["overall"]
        lines.append(
            _row(
                c["label"],
                f"{_f(s['precision'])} / {_f(p['precision'])}",
                f"{_f(s['recall'])} / {_f(p['recall'])}",
                _f(s["f1"]),
                _f(p["f1"]),
                _pct(c["untyped"]["recall"]),
                _negatives(c),
                _latency(c),
            )
        )
    lines += ["", "Precision and recall cells: strict / partial. Latency: mean / p95 per document.", ""]

    holdouts = [c for c in detection["configs"] if c.get("holdout")]
    if holdouts:
        documents = holdouts[0]["holdout"]["documents"]
        lines += [f"Held-out split only ({documents} documents written after the detectors were frozen):", ""]
        lines += _table(
            ["Configuration", "Precision", "Recall", "F1 partial", "F1 strict", "Any-type recall"], "lrrrrr"
        )
        for c in holdouts:
            h = c["holdout"]
            p = h["partial"]["overall"]
            scores = (_f(p["precision"]), _f(p["recall"]), _f(p["f1"]), _f(h["strict"]["f1"]))
            lines.append(_row(c["label"], *scores, _pct(h["untyped"]["recall"])))
        lines.append("")

    configs = detection["configs"]
    lines += _table(["Entity", "Gold", *(f"{c['label']} P / R / F1" for c in configs)], "lr" + "r" * len(configs))
    for kind, count in gold["by_type"].items():
        cells = []
        for c in configs:
            prf = c["partial"]["by_type"].get(kind)
            cells.append("–" if prf is None else f"{_f(prf['precision'])} / {_f(prf['recall'])} / {_f(prf['f1'])}")
        lines.append(_row(kind, count, *cells))
    lines += ["", "Per-type cells use partial matching (same type, overlapping span)."]

    subset = detection.get("llm_subset")
    if subset:
        lines += [
            "",
            f"### With the LLM detector: every {_ordinal(subset['every'])} document ({subset['documents']} documents, "
            f"{subset['entities']} entities), all configurations on the same subset",
            "",
        ]
        header = ["Configuration", "Precision", "Recall", "F1 partial", "F1 strict", "Any-type recall"]
        lines += _table([*header, "Clean docs flagged", "Latency", "Detector errors"], "lrrrrrrrr")
        for c in subset["configs"]:
            p = c["partial"]["overall"]
            scores = (_f(p["precision"]), _f(p["recall"]), _f(p["f1"]), _f(c["strict"]["overall"]["f1"]))
            lines.append(
                _row(
                    c["label"], *scores, _pct(c["untyped"]["recall"]), _negatives(c), _latency(c), c["detector_errors"]
                )
            )

    if detection.get("ner_models"):
        lines += ["", "### NER model choice (patterns + model, whole gold set)", ""]
        header = ["Model", "PERSON F1", "ADDRESS F1", "ORGANIZATION F1", "Overall F1", "NER ms / doc", "Load s"]
        lines += _table(header, "lrrrrrr")
        for row in detection["ner_models"]:
            if "error" in row:
                lines.append(_row(f"`{row['model']}`", f"could not load: {row['error'][:60]}", "", "", "", "", ""))
                continue
            by_type = row["by_type"]
            f1 = [_f((by_type.get(kind) or {}).get("f1")) for kind in ("PERSON", "ADDRESS", "ORGANIZATION")]
            ms = f"{row['ner_ms_mean'] or 0:.0f}"
            lines.append(_row(f"`{row['model']}`", *f1, _f(row["partial_f1"]), ms, row["load_seconds"]))
    return lines


def leak_tables(leak: dict[str, Any]) -> list[str]:
    lines = ["", "## Leak test (gold set through the gateway, recording upstream)", ""]
    header = ["Policy", "Detectors", "Channel", "Protected values", "Leaked", "Leak rate"]
    lines += _table([*header, "Partial name leaks", "Blocked requests"], "lllrrrrr")
    for r in leak["results"]:
        lines.append(
            _row(
                f"`{r['policy']}`",
                r["detectors"],
                r["channel"],
                r["protected_values"],
                r["leaked"],
                f"**{_pct(r['leak_rate'], 2)}**",
                r["leaked_partial_names"],
                f"{r['blocked_requests']}/{r['requests']}",
            )
        )
    return lines


def utility_tables(utility: dict[str, Any]) -> list[str]:
    verdicts = utility["verdicts"]
    return [
        "",
        "## Utility (real models)",
        "",
        f"Task `{utility['task']}`, policy `{utility['policy']}`, {utility['documents']} documents. "
        f"Model `{utility['model']}`, judge `{utility['judge_model']}`.",
        "",
        *_table(["Redacted vs original", "Count"], "lr"),
        *(_row(key, verdicts.get(key, 0)) for key in ("better", "equivalent", "worse", "error")),
        "",
        f"- Equivalent or better: **{_pct(utility['equivalent_or_better'])}**",
        f"- Restore accuracy: **{_pct(utility['restore_accuracy'])}** ({utility['placeholders_total']} placeholders "
        f"in raw answers, {utility['unknown_placeholders']} unknown, "
        f"{utility['leftover_placeholders']} left after restore)",
        f"- Name fidelity: **{_pct(utility['name_fidelity'])}** of {utility['name_fidelity_docs']} answers "
        "where the baseline used the customer's name",
        f"- Gold PII values sent upstream during the utility run: {utility['leaked_values']}",
        *_noise_floor(utility),
    ]


def _noise_floor(utility: dict[str, Any]) -> list[str]:
    noise = utility.get("noise_floor")
    if not noise:
        return []
    v = noise["verdicts"]
    return [
        f"- Noise floor (two samples of the original-text answer, same judge): equivalent or better "
        f"**{_pct(noise['equivalent_or_better'])}** (better {v.get('better', 0)}, equivalent {v.get('equivalent', 0)}, "
        f"worse {v.get('worse', 0)}, errors {v.get('error', 0)})",
    ]


def render_report(directory: Path) -> str:
    lines = ["# PII Shield evaluation report", ""]
    detection = _load(directory / "detection.json")
    leak = _load(directory / "leak.json")
    utility = _load(directory / "utility.json")
    calls = _load(directory / "calls_summary.json")
    if detection:
        lines += [f"Detection run: {detection['created']}.", "", *detection_tables(detection)]
    if leak:
        lines += leak_tables(leak)
    if utility:
        lines += utility_tables(utility)
    if calls:
        lines += [
            "",
            "## Real API calls",
            "",
            f"{calls['total_requests']} requests; every model id ends in `:free`: {calls['all_model_ids_free']}.",
            f"By tag: {calls['by_tag']}. By status: {calls['by_status']}. Served models: {calls['served_models']}.",
        ]
    return "\n".join(lines) + "\n"


def write_report(directory: Path) -> Path:
    path = directory / "report.md"
    path.write_text(render_report(directory), encoding="utf-8")
    return path
