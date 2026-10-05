"""Dashboard: playground (four panels), evaluation results and the audit log. Jinja2 + htmx, no build step."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pii_shield.anonymize.restore import PLACEHOLDER_PATTERN
from pii_shield.dashboard.examples import EXAMPLES, TASKS
from pii_shield.gateway.messages import add_placeholder_hint
from pii_shield.gateway.runtime import Runtime
from pii_shield.providers.base import ProviderError
from pii_shield.shield import RedactionResult, ShieldSession, new_session_id

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@dataclass
class Segment:
    text: str
    type: str | None = None
    title: str = ""


@dataclass
class PlaygroundResult:
    policy: str
    provider: str
    task: str
    original: list[Segment]
    sent: list[Segment]
    raw_answer: list[Segment] = field(default_factory=list)
    restored: list[Segment] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    blocked: bool = False
    block_reasons: list[str] = field(default_factory=list)
    error: str | None = None
    model: str | None = None
    redact_ms: float = 0.0
    llm_ms: float | None = None
    restored_count: int = 0
    unknown: list[str] = field(default_factory=list)
    detector_errors: dict[str, str] = field(default_factory=dict)


def original_segments(text: str, result: RedactionResult) -> list[Segment]:
    segments, cursor = [], 0
    for entity in result.entities:
        segments.append(Segment(text[cursor : entity.start]))
        title = f"{entity.type.value} · {entity.score:.2f} · {entity.source} → {entity.action.value}"
        segments.append(Segment(text[entity.start : entity.end], entity.type.value, title))
        cursor = entity.end
    segments.append(Segment(text[cursor:]))
    return [s for s in segments if s.text]


def sent_segments(text: str, result: RedactionResult) -> list[Segment]:
    """The redacted text, with each replacement highlighted (rebuilt in the same order it was built)."""
    segments, cursor = [], 0
    for entity in result.entities:
        segments.append(Segment(text[cursor : entity.start]))
        segments.append(Segment(entity.replacement, entity.type.value, f"{entity.type.value} → {entity.action.value}"))
        cursor = entity.end
    segments.append(Segment(text[cursor:]))
    return [s for s in segments if s.text]


def answer_segments(answer: str, session: ShieldSession, *, restored: bool) -> list[Segment]:
    """Split an answer on placeholders (and synthetic stand-ins); optionally swap in the originals."""
    synthetic = session.state.synthetic_map()
    segments: list[Segment] = []
    cursor = 0
    for match in PLACEHOLDER_PATTERN.finditer(answer):
        kind = (match.group(1) or match.group(3)).upper()
        entry = session.state.by_placeholder(f"<{kind}_{int(match.group(2) or match.group(4))}>")
        segments.append(Segment(answer[cursor : match.start()]))
        if entry is None:
            segments.append(Segment(match.group(0), "UNKNOWN", "placeholder not in this session"))
        else:
            segments.append(Segment(entry.original if restored else match.group(0), kind, f"{match.group(0)} ↔ {kind}"))
        cursor = match.end()
    segments.append(Segment(answer[cursor:]))
    if synthetic:
        segments = _split_synthetic(segments, synthetic, session, restored=restored)
    return [s for s in segments if s.text]


def _split_synthetic(
    segments: list[Segment], synthetic: dict[str, str], session: ShieldSession, *, restored: bool
) -> list[Segment]:
    kinds = {surface: entry.type.value for entry in session.state.entries for surface in entry.synthetic_variants}
    out: list[Segment] = []
    for segment in segments:
        if segment.type is not None:
            out.append(segment)
            continue
        text = segment.text
        while text:
            hits = [(text.find(s), s) for s in synthetic if s and text.find(s) >= 0]
            if not hits:
                out.append(Segment(text))
                break
            position, surface = min(hits, key=lambda hit: (hit[0], -len(hit[1])))
            out.append(Segment(text[:position]))
            shown = synthetic[surface] if restored else surface
            out.append(Segment(shown, kinds.get(surface), f"synthetic {surface!r}"))
            text = text[position + len(surface) :]
    return out


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def register_dashboard(app: FastAPI, runtime: Runtime) -> None:
    def context(request: Request, page: str, **extra: Any) -> dict[str, Any]:
        return {
            "request": request,
            "page": page,
            "upstream": runtime.upstream.label,
            "ner_status": runtime.ner_status,
            "policies": runtime.shield.policies.names,
            "default_policy": runtime.shield.policies.default_name,
            **extra,
        }

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def playground_page(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "playground.html",
            context(
                request,
                "playground",
                examples=EXAMPLES,
                tasks=TASKS,
                providers=list(runtime.playground_providers),
                text=EXAMPLES[0]["text"],
                policy=runtime.shield.policies.default_name,
                provider="fake",
                task="reply",
                result=None,
            ),
        )

    @app.post("/playground", response_class=HTMLResponse, include_in_schema=False)
    async def playground_run(
        request: Request,
        text: Annotated[str, Form()],
        policy: Annotated[str, Form()],
        provider: Annotated[str, Form()] = "fake",
        task: Annotated[str, Form()] = "reply",
    ) -> HTMLResponse:
        result = await run_playground(runtime, text, policy, provider, task)
        partial = request.headers.get("HX-Request") == "true"
        template = "partials/result.html" if partial else "playground.html"
        return TEMPLATES.TemplateResponse(
            request,
            template,
            context(
                request,
                "playground",
                examples=EXAMPLES,
                tasks=TASKS,
                providers=list(runtime.playground_providers),
                text=text,
                policy=policy,
                provider=provider,
                task=task,
                result=result,
            ),
        )

    @app.get("/results", response_class=HTMLResponse, include_in_schema=False)
    async def results_page(request: Request) -> HTMLResponse:
        directory = runtime.settings.results_dir
        detection = _load_json(directory / "detection.json")
        leak = _load_json(directory / "leak.json")
        utility = _load_json(directory / "utility.json")
        calls = _load_json(directory / "calls_summary.json")
        return TEMPLATES.TemplateResponse(
            request,
            "results.html",
            context(
                request,
                "results",
                detection=detection,
                leak=leak,
                utility=utility,
                calls=calls,
                results_dir=str(directory),
            ),
        )

    @app.get("/audit-log", response_class=HTMLResponse, include_in_schema=False)
    async def audit_page(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "audit.html",
            context(
                request,
                "audit",
                records=runtime.audit.recent(200),
                summary=runtime.audit.summary(),
                audit_file=str(runtime.settings.audit_file) if runtime.settings.audit_file else None,
            ),
        )


async def run_playground(runtime: Runtime, text: str, policy: str, provider_key: str, task: str) -> PlaygroundResult:
    session_id = new_session_id()
    policy_name = policy if policy in runtime.shield.policies.names else runtime.shield.policies.default_name

    def redact() -> tuple[RedactionResult, ShieldSession]:
        with runtime.shield.session(session_id, policy_name) as session:
            return session.redact(text), session

    result, session = await run_in_threadpool(redact)
    view = PlaygroundResult(
        policy=policy_name,
        provider=provider_key,
        task=task,
        original=original_segments(text, result),
        sent=sent_segments(text, result),
        entities=[
            {
                "type": e.type.value,
                "text": e.original,
                "score": e.score,
                "source": e.source,
                "action": e.action.value,
                "replacement": e.replacement,
            }
            for e in result.entities
        ],
        blocked=result.blocked,
        block_reasons=result.block_reasons,
        redact_ms=result.timings_ms.get("total", 0.0),
        detector_errors=result.detector_errors,
    )
    record = runtime.audit.new_record(route="playground", result=result, request_body=text, tenant=None)
    if result.blocked:
        record.status = 400
        runtime.audit.add(record)
        return view
    provider = runtime.playground_providers.get(provider_key) or runtime.playground_providers["fake"]
    record.upstream = provider.label
    instruction = TASKS.get(task, TASKS["reply"])["prompt"]
    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": result.text}]
    body = add_placeholder_hint({"model": "auto", "messages": messages, "max_tokens": 3000}, [result])
    started = time.perf_counter()
    try:
        answer = await provider.complete(body)
    except ProviderError as exc:
        view.error = str(exc)
        record.status = exc.status_code
        runtime.audit.add(record)
        return view
    view.llm_ms = round((time.perf_counter() - started) * 1000, 1)
    view.model = answer.get("model")
    raw = ((answer.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    _, report = session.restore_with_report(raw)
    view.raw_answer = answer_segments(raw, session, restored=False)
    view.restored = answer_segments(raw, session, restored=True)
    view.restored_count = report.restored + report.synthetic_restored
    view.unknown = report.unknown
    record.upstream_model = view.model
    record.upstream_ms = view.llm_ms
    record.restored = view.restored_count
    record.unknown_placeholders = len(report.unknown)
    record.total_ms = round(view.redact_ms + (view.llm_ms or 0.0), 1)
    runtime.audit.add(record)
    return view
