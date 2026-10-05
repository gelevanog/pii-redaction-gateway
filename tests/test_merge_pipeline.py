from pii_shield.detect.merge import resolve_overlaps, trim_span
from pii_shield.detect.names import name_variant_spans
from pii_shield.detect.pipeline import DetectionPipeline
from pii_shield.entities import EntityType, Span
from pii_shield.policy import Policy

from .conftest import StubNer


def span(
    text: str, value: str, kind: EntityType, score: float = 0.9, *, validated: bool = False, source: str = "t"
) -> Span:
    start = text.index(value)
    return Span(
        start=start, end=start + len(value), type=kind, text=value, score=score, validated=validated, source=source
    )


def test_validated_span_beats_unvalidated_overlap() -> None:
    text = "mail anna.petrova@gmail.com now"
    email = span(text, "anna.petrova@gmail.com", EntityType.EMAIL, 0.99, validated=True)
    person = span(text, "anna.petrova", EntityType.PERSON, 0.95)
    assert resolve_overlaps(text, [person, email]) == [email]


def test_same_type_overlaps_are_merged() -> None:
    text = "Hi Anna Petrova!"
    merged = resolve_overlaps(
        text, [span(text, "Anna", EntityType.PERSON), span(text, "Anna Petrova", EntityType.PERSON, 0.7)]
    )
    assert [(s.text, s.score) for s in merged] == [("Anna Petrova", 0.9)]


def test_higher_score_wins_between_types() -> None:
    text = "Ship to Morgan Street 5"
    address = span(text, "Morgan Street 5", EntityType.ADDRESS, 0.8)
    person = span(text, "Morgan", EntityType.PERSON, 0.6)
    assert resolve_overlaps(text, [person, address]) == [address]


def test_trim_titles_possessive_and_punctuation() -> None:
    text = "ask Ms. Petrova's assistant"
    trimmed = trim_span(text, span(text, "Ms. Petrova's", EntityType.PERSON))
    assert trimmed is not None and trimmed.text == "Petrova"
    assert trim_span("Да", Span(start=0, end=2, type=EntityType.PERSON, text="Да")) is None


def test_pattern_spans_are_not_trimmed() -> None:
    text = "call (415) 555-2671"
    phone = span(text, "(415) 555-2671", EntityType.PHONE)
    assert trim_span(text, phone) == phone


def test_name_variants_propagate_but_not_ambiguous_words() -> None:
    text = "Will Turner called. Will the fix ship? Mr. Turner says yes."
    person = span(text, "Will Turner", EntityType.PERSON)
    extra = name_variant_spans(text, [person.text], [person], score=0.85, source="name_variant")
    assert [s.text for s in extra] == ["Turner"]


def test_pipeline_thresholds_allow_and_deny_lists(support_policy: Policy) -> None:
    pipeline = DetectionPipeline()
    text = "Write to support@brightloop.com or anna@gmail.com about Project Falcon. Order 4155550132."
    found = {(s.type, s.text) for s in pipeline.detect(text, support_policy).spans}
    assert (EntityType.EMAIL, "anna@gmail.com") in found
    assert (EntityType.CUSTOM, "Project Falcon") in found
    assert all(value != "support@brightloop.com" for _, value in found)  # allow-list
    assert all(kind is not EntityType.PHONE for kind, _ in found)  # order-number context is below the threshold


def test_pipeline_ignores_existing_placeholders(support_policy: Policy) -> None:
    outcome = DetectionPipeline(ner=StubNer(people=["PERSON_1"])).detect(
        "Use <PERSON_1> and <EMAIL_2> as given.", support_policy
    )
    assert outcome.spans == []


def test_pipeline_reports_missing_ner_only_when_expected(support_policy: Policy) -> None:
    assert DetectionPipeline().detect("hello", support_policy).errors == {}
    assert "ner" in DetectionPipeline(ner_expected=True).detect("hello", support_policy).errors


def test_pipeline_reports_failing_detector(support_policy: Policy) -> None:
    class Broken:
        name = "ner"

        def detect(self, text: str) -> list[Span]:
            raise RuntimeError("model crashed")

    outcome = DetectionPipeline(ner=Broken(), ner_expected=True).detect("Anna", support_policy)
    assert outcome.errors["ner"].startswith("RuntimeError")


def test_pipeline_uses_known_people_from_session(support_policy: Policy) -> None:
    outcome = DetectionPipeline().detect("Petrova called again", support_policy, known_people=["Anna Petrova"])
    assert [(s.type, s.text, s.source) for s in outcome.spans] == [(EntityType.PERSON, "Petrova", "name_variant")]


def test_role_nouns_are_never_people() -> None:
    text = "You are a customer-support agent. Address the customer by name; the IT manager agrees."
    spans = [span(text, "customer-support agent", EntityType.PERSON), span(text, "IT manager", EntityType.PERSON)]
    assert resolve_overlaps(text, spans) == []
    assert trim_span("Agent Smith", Span(start=0, end=11, type=EntityType.PERSON, text="Agent Smith")) is not None
