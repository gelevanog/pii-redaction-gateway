"""OpenAI-compatible PII gateway (FastAPI).

Point any OpenAI SDK at it (`base_url="http://localhost:8000/v1"`): requests are redacted, forwarded to
the configured upstream, and answers are restored. Streaming (SSE) is supported. Pick a policy per
route (`/p/strict-finance/v1`), per header (`X-PII-Policy`) or per tenant API key.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from pii_shield import __version__
from pii_shield.audit import AuditRecord
from pii_shield.config import Settings
from pii_shield.gateway.messages import (
    StreamTransformer,
    add_placeholder_hint,
    merge_results,
    redact_chat_request,
    restore_chat_response,
)
from pii_shield.gateway.runtime import Runtime, Tenant, build_runtime
from pii_shield.logging_config import configure_logging, get_logger
from pii_shield.policy import PolicyError
from pii_shield.providers.base import JsonDict, ProviderError
from pii_shield.shield import RedactionResult, ShieldSession, new_session_id

log = get_logger(__name__)

SESSION_HEADER = "X-PII-Session-Id"
POLICY_HEADER = "X-PII-Policy"


def openai_error(status: int, message: str, error_type: str, code: str | None = None, **extra: Any) -> JSONResponse:
    error: JsonDict = {"message": message, "type": error_type, "param": None, "code": code, **extra}
    return JSONResponse({"error": error}, status_code=status)


class RedactRequest(BaseModel):
    text: str
    policy: str | None = None
    session_id: str | None = None


class RestoreRequest(BaseModel):
    text: str
    session_id: str


def block_message(policy: str, reasons: list[str]) -> str:
    entities = [r for r in reasons if not r.startswith("detector_unavailable:")]
    detectors = [r.split(":", 1)[1] for r in reasons if r.startswith("detector_unavailable:")]
    parts = []
    if entities:
        parts.append(f"it contains {', '.join(entities)}; remove that data and try again")
    if detectors:
        parts.append(f"detector(s) {', '.join(detectors)} unavailable and the policy is fail-closed")
    return f"Request refused by PII policy {policy!r}: " + "; ".join(parts) + "."


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def resolve_tenant(runtime: Runtime, authorization: str | None) -> Tenant | None:
    if not runtime.tenants.enabled:
        return None
    tenant = runtime.tenants.lookup(_bearer(authorization))
    if tenant is None:
        raise HTTPException(status_code=401, detail="invalid or missing gateway API key")
    return tenant


def resolve_policy(runtime: Runtime, tenant: Tenant | None, route_policy: str | None, header_policy: str | None) -> str:
    requested = route_policy or (header_policy if runtime.settings.allow_policy_header else None)
    if tenant is not None:
        if requested and requested != tenant.policy and requested not in tenant.allowed_policies:
            raise HTTPException(status_code=403, detail=f"tenant {tenant.name!r} may not use policy {requested!r}")
        requested = requested or tenant.policy
    name = requested or runtime.settings.default_policy
    try:
        runtime.shield.policies.get(name)
    except PolicyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return name


def redact_request(
    runtime: Runtime, body: JsonDict, policy: str, session_id: str
) -> tuple[JsonDict, RedactionResult, ShieldSession]:
    """Runs in a worker thread: NER is CPU-bound and the Redis vault client is synchronous."""
    with runtime.shield.session(session_id, policy) as session:
        redacted, results = redact_chat_request(session, body)
    if runtime.settings.placeholder_hint:
        redacted = add_placeholder_hint(redacted, results)
    return redacted, merge_results(results, session_id, policy), session


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or (runtime.settings if runtime else Settings())
    configure_logging(settings.log_level, settings.log_format)
    runtime = runtime or build_runtime(settings)
    app = FastAPI(
        title="PII Shield",
        version=__version__,
        description="Strip personal data before it reaches an LLM, and put it back in the answer.",
    )
    app.state.runtime = runtime

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        # OpenAI SDKs read `error.message`; keep FastAPI's `detail` for everything else.
        kind = {401: "authentication_error", 403: "permission_error"}.get(exc.status_code, "invalid_request_error")
        body = {"error": {"message": str(exc.detail), "type": kind, "param": None, "code": None}, "detail": exc.detail}
        return JSONResponse(body, status_code=exc.status_code)

    async def chat_completions(
        request: Request,
        body: JsonDict,
        route_policy: str | None,
        authorization: str | None,
        policy_header: str | None,
        session_header: str | None,
    ) -> Any:
        started = time.perf_counter()
        tenant = resolve_tenant(runtime, authorization)
        policy = resolve_policy(runtime, tenant, route_policy, policy_header)
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            return openai_error(400, "`messages` must be a non-empty list", "invalid_request_error")
        session_id = session_header or new_session_id()
        raw = await request.body()
        try:
            redacted, summary, session = await run_in_threadpool(redact_request, runtime, body, policy, session_id)
        except TimeoutError as exc:
            return openai_error(409, str(exc), "session_busy")
        record = runtime.audit.new_record(
            route="chat.completions", result=summary, request_body=raw, tenant=tenant.name if tenant else None
        )
        record.upstream = runtime.upstream.label
        record.stream = bool(body.get("stream"))
        headers = {SESSION_HEADER: session_id, POLICY_HEADER: policy, "X-PII-Entities": str(len(summary.entities))}

        if summary.blocked:
            record.status = 400
            record.total_ms = _ms(started)
            runtime.audit.add(record)
            log.info("request.blocked", policy=policy, reasons=summary.block_reasons)
            response = openai_error(
                400,
                block_message(policy, summary.block_reasons),
                "pii_policy_violation",
                "pii_blocked",
                pii_entities=summary.block_reasons,
            )
            response.headers.update(headers)
            return response

        if body.get("stream"):
            return StreamingResponse(
                _stream(runtime, redacted, session, record, started),
                media_type="text/event-stream",
                headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        upstream_started = time.perf_counter()
        try:
            answer = await runtime.upstream.complete(redacted)
        except ProviderError as exc:
            record.status = exc.status_code
            record.total_ms = _ms(started)
            runtime.audit.add(record)
            log.warning("upstream.error", status=exc.status_code, error=str(exc)[:200])
            response = openai_error(exc.status_code, str(exc), "upstream_error")
            response.headers.update(headers)
            return response
        record.upstream_ms = _ms(upstream_started)
        restored, report = restore_chat_response(session, answer)
        restored.pop("_cached", None)
        record.upstream_model = answer.get("model")
        record.restored = report.restored + report.synthetic_restored
        record.unknown_placeholders = len(report.unknown)
        record.total_ms = _ms(started)
        runtime.audit.add(record)
        log.info("request.done", policy=policy, entities=len(summary.entities), total_ms=record.total_ms)
        return JSONResponse(restored, headers=headers)

    @app.post("/v1/chat/completions", tags=["OpenAI-compatible"])
    async def chat(
        request: Request,
        body: Annotated[JsonDict, Body()],
        authorization: Annotated[str | None, Header()] = None,
        x_pii_policy: Annotated[str | None, Header()] = None,
        x_pii_session_id: Annotated[str | None, Header()] = None,
    ) -> Any:
        return await chat_completions(request, body, None, authorization, x_pii_policy, x_pii_session_id)

    @app.post("/p/{policy}/v1/chat/completions", tags=["OpenAI-compatible"])
    async def chat_with_policy(
        policy: str,
        request: Request,
        body: Annotated[JsonDict, Body()],
        authorization: Annotated[str | None, Header()] = None,
        x_pii_session_id: Annotated[str | None, Header()] = None,
    ) -> Any:
        return await chat_completions(request, body, policy, authorization, None, x_pii_session_id)

    @app.get("/v1/models", tags=["OpenAI-compatible"])
    @app.get("/p/{policy}/v1/models", tags=["OpenAI-compatible"], include_in_schema=False)
    async def models(policy: str | None = None) -> JsonDict:
        upstream = runtime.upstream
        model_ids = [getattr(upstream, "default_model", None) or getattr(upstream, "model", None) or upstream.label]
        model_ids += list(getattr(upstream, "fallback_models", []))
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "pii-shield"} for m in model_ids]}

    @app.post("/v1/redact", tags=["PII"])
    async def redact(payload: RedactRequest, authorization: Annotated[str | None, Header()] = None) -> JsonDict:
        """Redact one text (library over HTTP). Returns entity positions and types, never the original values."""
        tenant = resolve_tenant(runtime, authorization)
        policy = resolve_policy(runtime, tenant, None, payload.policy)
        session_id = payload.session_id or new_session_id()
        result: RedactionResult = await run_in_threadpool(
            runtime.shield.redact, payload.text, policy=policy, session_id=session_id
        )
        record = runtime.audit.new_record(
            route="redact", result=result, request_body=payload.text, tenant=tenant.name if tenant else None
        )
        record.total_ms = result.timings_ms.get("total", 0.0)
        runtime.audit.add(record)
        return {
            "text": result.text,
            "session_id": result.session_id,
            "policy": result.policy,
            "blocked": result.blocked,
            "block_reasons": result.block_reasons,
            "entities": [e.model_dump(mode="json", exclude={"original"}) for e in result.entities],
            "timings_ms": result.timings_ms,
        }

    @app.post("/v1/restore", tags=["PII"])
    async def restore(payload: RestoreRequest, authorization: Annotated[str | None, Header()] = None) -> JsonDict:
        resolve_tenant(runtime, authorization)

        def run() -> JsonDict:
            with runtime.shield.session(payload.session_id) as session:
                text, report = session.restore_with_report(payload.text)
            return {
                "text": text,
                "restored": report.restored + report.synthetic_restored,
                "unknown_placeholders": report.unknown,
            }

        return await run_in_threadpool(run)

    @app.get("/audit", tags=["PII"])
    async def audit(limit: int = 100, authorization: Annotated[str | None, Header()] = None) -> JsonDict:
        """Recent requests: entity counts, policy, latency, request hash. No values, ever."""
        resolve_tenant(runtime, authorization)
        records = runtime.audit.recent(min(max(limit, 1), 1000))
        return {"summary": runtime.audit.summary(), "records": [r.model_dump(mode="json") for r in records]}

    @app.get("/health", tags=["ops"])
    async def health() -> JsonDict:
        shield = runtime.shield
        return {
            "status": "ok",
            "version": __version__,
            "upstream": runtime.upstream.label,
            "require_free_models": settings.require_free_models,
            "policies": shield.policies.names,
            "default_policy": shield.policies.default_name,
            "detectors": shield.pipeline.available(),
            "ner": runtime.ner_status,
            "vault": {
                "backend": shield.vault.backend.name,
                "ttl_seconds": shield.vault.ttl_seconds,
                "ephemeral_key": runtime.ephemeral_key,
            },
            "tenants": runtime.tenants.enabled,
        }

    from pii_shield.dashboard.views import register_dashboard

    register_dashboard(app, runtime)
    return app


async def _stream(
    runtime: Runtime, body: JsonDict, session: ShieldSession, record: AuditRecord, started: float
) -> AsyncIterator[str]:
    transformer = StreamTransformer(session)
    upstream_started = time.perf_counter()
    try:
        async for chunk in runtime.upstream.stream(body):
            if record.upstream_model is None and chunk.get("model"):
                record.upstream_model = chunk["model"]
            yield f"data: {json.dumps(transformer.transform(chunk), ensure_ascii=False)}\n\n"
        tail = transformer.finish()
        if tail is not None:
            yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except ProviderError as exc:
        record.status = exc.status_code
        log.warning("upstream.stream_error", status=exc.status_code, error=str(exc)[:200])
        error = {"error": {"message": str(exc), "type": "upstream_error", "code": exc.status_code}}
        yield f"data: {json.dumps(error)}\n\n"
    finally:
        record.upstream_ms = _ms(upstream_started)
        record.restored = transformer.report.restored + transformer.report.synthetic_restored
        record.unknown_placeholders = len(transformer.report.unknown)
        record.total_ms = _ms(started)
        runtime.audit.add(record)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def create_default_app() -> FastAPI:
    """Factory for `uvicorn pii_shield.gateway.app:create_default_app --factory`."""
    return create_app(Settings())
