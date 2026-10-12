"""Re-judge the committed free-model utility answers (results/utility.json) with the paid runs' judge.

The free baseline was judged by a free model; the paid runs by claude-opus-5-5. Re-judging the same free
answers with the same Opus judge separates "better answer model" from "different judge".
No answers are regenerated: only the judge is called, through the eval's own judge prompt and verdict logic.

    OPENAI_API_KEY=<endpoint key> PII_SHIELD_ALLOW_PAID_MODELS=1 \
      uv run python results/paid-models/rejudge_free_baseline.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pii_shield.config import Settings
from pii_shield.eval.config import load_eval_config
from pii_shield.eval.gold import load_gold
from pii_shield.eval.utility import UtilityItem, _judge
from pii_shield.providers.factory import build_provider
from pii_shield.providers.resilient import CallLedger, DiskCache, ResilientProvider, Throttle

OUT = Path("results/paid-models/free-baseline-opus-judge")


def main() -> None:
    config = load_eval_config("configs/paid-models/haiku.yaml")  # only its judge settings are used
    settings = Settings()
    if not settings.allow_paid_models:
        raise SystemExit("set PII_SHIELD_ALLOW_PAID_MODELS=1 to call the paid judge")
    judge_cfg = config.utility.judge
    judge = ResilientProvider(
        build_provider(settings, kind=judge_cfg.provider, model=judge_cfg.model, base_url=judge_cfg.base_url),
        ledger=CallLedger(OUT / "calls.jsonl", 40),
        cache=DiskCache(settings.llm_cache_dir),
        throttle=Throttle(config.min_seconds_between_requests),
        max_retries=1,
        tag="utility_judge",
    )
    free = json.loads(Path("results/utility.json").read_text(encoding="utf-8"))
    docs = {d.id: d for d in load_gold(config.gold)}
    items = [UtilityItem.model_validate(raw) for raw in free["items"]]

    async def judge_all() -> list[dict[str, str]]:
        rows = []
        for item in items:
            if item.error:
                continue
            try:
                verdict, reason = await _judge(judge, config.utility, docs[item.doc], item)
            except Exception as exc:  # report and continue: one failed judge call must not lose the others
                verdict, reason = "error", str(exc)[:200]
            rows.append(
                {"doc": item.doc, "free_judge_verdict": item.verdict, "opus_verdict": verdict, "reason": reason}
            )
        return rows

    rows = asyncio.run(judge_all())
    counts = {v: sum(1 for r in rows if r["opus_verdict"] == v) for v in ("better", "equivalent", "worse", "error")}
    judged = counts["better"] + counts["equivalent"] + counts["worse"]
    summary = {
        "source": "results/utility.json",
        "answer_model": free["model"],
        "original_judge": free["judge_model"],
        "judge_model": judge_cfg.model,
        "documents": len(rows),
        "verdicts": counts,
        "equivalent_or_better": (counts["better"] + counts["equivalent"]) / judged if judged else None,
        "agreement_with_original_judge": sum(r["free_judge_verdict"] == r["opus_verdict"] for r in rows) / len(rows),
        "items": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "utility_rejudged.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print({k: v for k, v in summary.items() if k != "items"})


if __name__ == "__main__":
    main()
