"""Run the evaluation stages and write small, committable artifacts to the output directory.

detection.json   P/R/F1 per type and detector configuration, latency, error examples
leak.json        share of gold PII values that reached the upstream, per policy / detectors / channel
utility.json     real-model answer quality with vs without redaction, restore accuracy
calls_summary.json, calls.jsonl   every real API request (ledger)
report.md        the tables used in the README
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pii_shield.config import Settings
from pii_shield.detect.llm import LlmDetector
from pii_shield.detect.ner import GlinerDetector
from pii_shield.detect.pipeline import DetectionPipeline
from pii_shield.eval.config import DetectorConfig, EvalConfig, LlmModelConfig
from pii_shield.eval.detection import ConfigResult, DetectionReport, SubsetReport, evaluate_config, new_report
from pii_shield.eval.gold import GoldDoc, load_gold
from pii_shield.eval.leak import LeakResult, run_leak_test
from pii_shield.eval.utility import UtilityReport, run_utility
from pii_shield.logging_config import get_logger
from pii_shield.policy import Policy, PolicySet
from pii_shield.providers.factory import build_provider
from pii_shield.providers.resilient import CallLedger, DiskCache, ResilientProvider, Throttle

log = get_logger(__name__)


class EvalRunner:
    def __init__(self, config: EvalConfig, settings: Settings) -> None:
        self.config = config
        self.settings = settings.model_copy(
            update={
                "require_free_models": config.require_free_models,
                "ner_enabled": False,  # the runner builds its own pipelines
                "llm_ledger": config.output_dir / "calls.jsonl",
            }
        )
        self.out = config.output_dir
        self.docs = load_gold(config.gold)
        self.policies = PolicySet.from_dir(self.settings.policies_dir, self.settings.default_policy)
        self._ner: dict[str, GlinerDetector] = {}
        self._ledger: CallLedger | None = None
        self._throttle = Throttle(config.min_seconds_between_requests)

    # ------------------------------------------------------------------------------------ helpers
    def ner(self, model: str | None = None) -> GlinerDetector:
        name = model or self.config.ner_model
        if name not in self._ner:
            detector = GlinerDetector(model_name=name, threads=self.settings.ner_threads)
            detector.load()
            self._ner[name] = detector
        return self._ner[name]

    def llm(self, llm: LlmModelConfig, tag: str) -> ResilientProvider:
        if self._ledger is None:
            self._ledger = CallLedger(self.out / "calls.jsonl", self.config.max_calls)
        inner = build_provider(self.settings, kind=llm.provider, model=llm.model, fallback_models=llm.fallback_models)
        return ResilientProvider(
            inner,
            ledger=self._ledger,
            cache=DiskCache(self.settings.llm_cache_dir),
            throttle=self._throttle,
            max_retries=self.settings.llm_max_retries,
            tag=tag,
        )

    def pipeline(self, *, ner: bool, llm: bool) -> DetectionPipeline:
        pipeline = DetectionPipeline(ner_expected=ner)
        if ner:
            pipeline.ner = self.ner()
        if llm:
            provider = self.llm(self.config.llm_detector, "llm_detector")
            pipeline.llm = LlmDetector(
                provider=provider,
                model=self.config.llm_detector.model,
                openrouter=self.config.llm_detector.provider == "openrouter",
            )
        return pipeline

    def _write(self, name: str, data: object) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / name
        text = (
            data.model_dump_json(indent=1)
            if hasattr(data, "model_dump_json")
            else json.dumps(data, indent=1, ensure_ascii=False)
        )
        path.write_text(text + "\n", encoding="utf-8")
        return path

    # ------------------------------------------------------------------------------------ stages
    def detection(self, only: list[str] | None = None) -> DetectionReport:
        """Offline configs on the whole gold set; LLM configs (and, for comparison, the others) on a subset."""
        policy = self.policies.get(self.config.policy)
        existing = self.out / "detection.json"
        report = new_report(self.docs, policy.name)
        if existing.exists():
            previous = DetectionReport.model_validate_json(existing.read_text(encoding="utf-8"))
            report.configs = [c for c in previous.configs if only and c.name not in only]
            report.ner_models = previous.ner_models
            report.llm_subset = previous.llm_subset
        subset = sorted(self.docs, key=lambda d: d.id)[:: self.config.llm_subset_every]
        selected = [c for c in self.config.detectors if not only or c.name in only]
        for config in selected:
            if config.llm:
                continue
            log.info("eval.detection", config=config.name, docs=len(self.docs))
            report.configs.append(self._evaluate(self.docs, policy, config))
        if any(c.llm for c in selected):
            results = []
            for config in self.config.detectors:
                log.info("eval.detection.subset", config=config.name, docs=len(subset))
                results.append(self._evaluate(subset, policy, config))
            report.llm_subset = SubsetReport(
                documents=len(subset),
                every=self.config.llm_subset_every,
                entities=sum(len(d.entities) for d in subset),
                configs=results,
            )
        order = {c.name: i for i, c in enumerate(self.config.detectors)}
        report.configs.sort(key=lambda c: order.get(c.name, 99))
        if not only or "ner-models" in only:
            report.ner_models = self._ner_models(policy)
        self._write("detection.json", report)
        return report

    def _evaluate(self, docs: list[GoldDoc], policy: Policy, config: DetectorConfig) -> ConfigResult:
        pipeline = self.pipeline(ner=config.ner, llm=config.llm)
        models = {}
        if config.ner:
            models["ner"] = self.config.ner_model
        if config.llm:
            models["llm"] = self.config.llm_detector.model
        workers = self.config.llm_concurrency if config.llm else 1
        return evaluate_config(docs, pipeline, policy, config, models=models, workers=workers)

    def _ner_models(self, policy: Policy) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for model in [self.config.ner_model, *self.config.ner_compare]:
            try:
                detector = self.ner(model)
            except Exception as exc:  # report a model that cannot load instead of failing the whole run
                rows.append({"model": model, "error": str(exc)[:200]})
                continue
            pipeline = DetectionPipeline(ner=detector, ner_expected=True)
            config = DetectorConfig(name=f"ner:{model}", label=model, ner=True)
            result = evaluate_config(self.docs, pipeline, policy, config)
            ner_types = {t: result.partial.by_type.get(t) for t in ("PERSON", "ADDRESS", "ORGANIZATION")}
            rows.append(
                {
                    "model": model,
                    "load_seconds": detector.load_seconds,
                    "partial_f1": result.partial.overall.f1,
                    "by_type": {t: (prf.model_dump() if prf else None) for t, prf in ner_types.items()},
                    "ner_ms_mean": result.layer_latency_ms.get("ner"),
                }
            )
        return rows

    def leak(self) -> list[LeakResult]:
        results: list[LeakResult] = []
        for name in self.config.leak_detectors:
            config = self.config.detector(name)
            pipeline = self.pipeline(ner=config.ner, llm=config.llm)
            log.info("eval.leak", detectors=name)
            results += asyncio.run(
                run_leak_test(self.docs, self.settings, pipeline, self.config.leak_policies, config.label)
            )
        self._write(
            "leak.json", {"created": _now(), "documents": len(self.docs), "results": [r.model_dump() for r in results]}
        )
        return results

    def utility(self) -> UtilityReport:
        cfg = self.config.utility
        pipeline = self.pipeline(ner=True, llm=False)
        model = self.llm(cfg.model, "utility_model")
        judge = self.llm(cfg.judge, "utility_judge")
        report = asyncio.run(run_utility(self.docs, cfg, self.settings, pipeline, model, judge, _now()))
        self._write("utility.json", report)
        return report

    def calls_summary(self) -> dict[str, object]:
        summary = summarize_ledger(self.out / "calls.jsonl")
        self._write("calls_summary.json", summary)
        return summary


def summarize_ledger(path: Path) -> dict[str, object]:
    rows = (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if path.exists()
        else []
    )
    models = Counter(str(r.get("served_model") or r.get("requested_model")) for r in rows if r.get("status") == "ok")
    every_id = {str(r.get("requested_model")) for r in rows if r.get("requested_model")} | {
        str(r.get("served_model")) for r in rows if r.get("served_model")
    }
    return {
        "total_requests": len(rows),
        "by_status": dict(Counter(r.get("status") for r in rows)),
        "by_tag": dict(Counter(r.get("tag") for r in rows)),
        "served_models": dict(models.most_common()),
        "all_model_ids_free": all(m.endswith(":free") for m in every_id if m not in {"None", "auto"}),
        "input_tokens": sum(int(r.get("input_tokens") or 0) for r in rows),
        "output_tokens": sum(int(r.get("output_tokens") or 0) for r in rows),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
