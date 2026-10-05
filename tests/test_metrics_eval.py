from pathlib import Path

import pytest

from pii_shield.config import Settings
from pii_shield.detect.pipeline import DetectionPipeline
from pii_shield.eval.config import DetectorConfig
from pii_shield.eval.detection import evaluate_config
from pii_shield.eval.gold import GoldDoc, GoldFormatError, build_gold, parse_markup
from pii_shield.eval.leak import run_leak_test
from pii_shield.eval.metrics import MatchMode, SpanRef, aggregate, percentile, score_document
from pii_shield.eval.report import render_report
from pii_shield.policy import Policy

GOLD = [SpanRef("PERSON", 0, 11), SpanRef("EMAIL", 20, 34)]


def test_strict_partial_untyped() -> None:
    predicted = [SpanRef("PERSON", 0, 4), SpanRef("ORGANIZATION", 20, 34), SpanRef("PHONE", 40, 50)]
    strict, _ = aggregate([score_document(GOLD, predicted, MatchMode.STRICT)])
    partial, by_type = aggregate([score_document(GOLD, predicted, MatchMode.PARTIAL)])
    untyped, _ = aggregate([score_document(GOLD, predicted, MatchMode.UNTYPED)])
    assert (strict.tp, strict.fp, strict.fn) == (0, 3, 2)
    assert (partial.tp, partial.fp, partial.fn) == (1, 2, 1) and by_type["EMAIL"].fn == 1
    assert (untyped.tp, untyped.fp, untyped.fn) == (2, 1, 0)
    assert untyped.recall == 1.0 and untyped.precision == pytest.approx(2 / 3)


def test_one_to_one_matching_and_empty() -> None:
    counts = score_document(
        [SpanRef("PERSON", 0, 10)], [SpanRef("PERSON", 0, 4), SpanRef("PERSON", 5, 10)], MatchMode.PARTIAL
    )
    assert (counts["PERSON"].tp, counts["PERSON"].fp) == (1, 1)
    overall, _ = aggregate([])
    assert overall.f1 is None
    assert percentile([1, 2, 3, 4], 0.5) == 2.5 and percentile([], 0.9) == 0.0


def test_gold_markup() -> None:
    text, entities = parse_markup("d1", "Hi [[Anna|PERSON]], mail [[a@b.co|EMAIL]].")
    assert text == "Hi Anna, mail a@b.co." and [(e.start, e.end) for e in entities] == [(3, 7), (14, 20)]
    with pytest.raises(GoldFormatError):
        parse_markup("d2", "[[Anna|NAME]]")


def test_committed_gold_set_builds_and_matches_jsonl() -> None:
    docs = build_gold(Path("data/gold/source"))
    assert len(docs) >= 150 and sum(len(d.entities) for d in docs) >= 300
    committed = Path("data/gold/gold.jsonl").read_text(encoding="utf-8").splitlines()
    assert [d.model_dump_json() for d in docs] == committed  # gold.jsonl is up to date with the source


def docs() -> list[GoldDoc]:
    doc1_text, doc1 = parse_markup("a", "Card [[4111 1111 1111 1111|CREDIT_CARD]] for [[Anna Petrova|PERSON]]")
    doc2_text, _ = parse_markup("b", "No personal data at all here.")
    return [
        GoldDoc(id="a", domain="t", lang="en", text=doc1_text, entities=doc1),
        GoldDoc(id="b", domain="t", lang="en", text=doc2_text),
    ]


def test_detection_eval_on_tiny_set(support_policy: Policy) -> None:
    result = evaluate_config(docs(), DetectionPipeline(), support_policy, DetectorConfig(name="p", label="Patterns"))
    assert result.partial.by_type["CREDIT_CARD"].tp == 1 and result.partial.by_type["PERSON"].fn == 1
    assert result.negatives == 1 and result.negatives_with_findings == 0
    assert [e.value for e in result.errors] == ["Anna Petrova"]


async def test_leak_test_counts_values_reaching_upstream(settings: Settings, pipeline: DetectionPipeline) -> None:
    results = await run_leak_test(docs(), settings, pipeline, ["support-chat", "strict-finance"], "stub")
    by_key = {(r.policy, r.channel): r for r in results}
    assert by_key[("support-chat", "user")].leaked == 0 and by_key[("support-chat", "tool_result")].leaked == 0
    assert by_key[("strict-finance", "user")].blocked_requests == 1  # the card blocks the whole request
    no_ner = await run_leak_test(
        docs(), settings, DetectionPipeline(), ["support-chat"], "patterns", channels=("user",)
    )
    assert no_ner[0].leaked == 1 and no_ner[0].examples[0].value == "Anna Petrova"


def test_report_renders_without_results(tmp_path: Path) -> None:
    assert render_report(tmp_path).startswith("# PII Shield evaluation report")
