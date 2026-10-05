"""The OpenAI-compatible endpoint with the fake upstream: non-stream, stream, tools, policies, block, tenants."""

import hashlib
import json
from pathlib import Path

import httpx
import openai
import pytest
from fastapi import FastAPI

from pii_shield.config import Settings
from pii_shield.detect.pipeline import DetectionPipeline
from pii_shield.gateway.app import create_app
from pii_shield.gateway.runtime import build_runtime
from pii_shield.providers.fake import RecordingProvider

MESSAGE = "Hi, I'm Anna Petrova (anna.petrova@gmail.com, +44 7911 123456). My card is 4111 1111 1111 1111."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "parameters": {
                "type": "object",
                "properties": {"to_email": {"type": "string"}, "customer_name": {"type": "string"}},
            },
        },
    }
]


def sdk(app: FastAPI, base: str = "http://gateway/v1", api_key: str = "unused") -> openai.AsyncOpenAI:
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    return openai.AsyncOpenAI(api_key=api_key, base_url=base, http_client=http)


def upstream_text(recorder: RecordingProvider) -> str:
    return json.dumps(recorder.requests[-1], ensure_ascii=False)


async def test_non_stream_redacts_upstream_and_restores_answer(app: FastAPI, recorder: RecordingProvider) -> None:
    raw = await sdk(app).chat.completions.with_raw_response.create(
        model="fake", messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": MESSAGE}]
    )
    answer = raw.parse().choices[0].message.content or ""
    sent = upstream_text(recorder)
    for value in ("Anna Petrova", "anna.petrova@gmail.com", "7911 123456", "4111 1111 1111 1111"):
        assert value not in sent
    assert "<PERSON_1>" in sent and "**** **** **** 1111" in sent
    assert answer.startswith("Hi Anna Petrova, thanks")
    assert "anna.petrova@gmail.com" in answer and "<EMAIL_1>" not in answer
    assert raw.headers["x-pii-session-id"] and raw.headers["x-pii-policy"] == "support-chat"


async def test_stream_restores_split_placeholders(app: FastAPI, recorder: RecordingProvider) -> None:
    stream = await sdk(app).chat.completions.create(
        model="fake", messages=[{"role": "user", "content": MESSAGE}], stream=True
    )
    text = ""
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    assert text.startswith("Hi Anna Petrova, thanks") and "+44 7911 123456" in text
    assert "<PHONE_1>" in upstream_text(recorder)


@pytest.mark.parametrize("stream", [False, True])
async def test_tool_call_arguments_restored_for_the_client(
    app: FastAPI, recorder: RecordingProvider, stream: bool
) -> None:
    client = sdk(app)
    messages = [{"role": "user", "content": MESSAGE}]
    if stream:
        arguments = ""
        async for chunk in await client.chat.completions.create(
            model="fake", messages=messages, tools=TOOLS, stream=True
        ):
            for call in (chunk.choices[0].delta.tool_calls or []) if chunk.choices else []:
                arguments += call.function.arguments or "" if call.function else ""
    else:
        response = await client.chat.completions.create(model="fake", messages=messages, tools=TOOLS)
        arguments = response.choices[0].message.tool_calls[0].function.arguments  # type: ignore[index,union-attr]
    assert json.loads(arguments) == {"to_email": "anna.petrova@gmail.com", "customer_name": "Anna Petrova"}
    assert "<EMAIL_1>" in upstream_text(recorder)


async def test_tool_messages_and_arguments_are_redacted(client: httpx.AsyncClient, recorder: RecordingProvider) -> None:
    body = {
        "model": "fake",
        "messages": [
            {"role": "user", "content": "Look up the customer.", "name": "agent_jane"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": json.dumps({"email": "anna.petrova@gmail.com", "n": 3}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "Customer Anna Petrova, IBAN DE89 3704 0044 0532 0130 00",
            },
        ],
    }
    response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    hint, *sent = recorder.requests[-1]["messages"]
    assert hint["role"] == "system" and "placeholders" in hint["content"]  # added by the gateway
    assert json.loads(sent[1]["tool_calls"][0]["function"]["arguments"]) == {"email": "<EMAIL_1>", "n": 3}
    assert sent[2]["content"] == "Customer <PERSON_1>, IBAN **** **** **** **** **30 00"
    assert "name" not in sent[0]
    assert "Anna Petrova" in response.json()["choices"][0]["message"]["content"]


async def test_block_policy_refuses_and_never_calls_upstream(app: FastAPI, recorder: RecordingProvider) -> None:
    with pytest.raises(openai.BadRequestError) as excinfo:
        await sdk(app, "http://gateway/p/strict-finance/v1").chat.completions.create(
            model="fake", messages=[{"role": "user", "content": MESSAGE}]
        )
    assert excinfo.value.code == "pii_blocked"
    assert "CREDIT_CARD" in str(excinfo.value)
    assert recorder.requests == []


async def test_policy_header_and_unknown_policy(client: httpx.AsyncClient) -> None:
    body = {"model": "fake", "messages": [{"role": "user", "content": "Mail anna@gmail.com"}]}
    hashed = await client.post("/v1/chat/completions", json=body, headers={"X-PII-Policy": "analytics-irreversible"})
    assert hashed.headers["x-pii-policy"] == "analytics-irreversible"
    assert "<EMAIL:" in hashed.json()["choices"][0]["message"]["content"]  # hashes are never restored
    unknown = await client.post("/v1/chat/completions", json=body, headers={"X-PII-Policy": "nope"})
    assert unknown.status_code == 400 and "unknown policy" in unknown.json()["error"]["message"]


async def test_session_header_keeps_placeholders_across_requests(
    client: httpx.AsyncClient, recorder: RecordingProvider
) -> None:
    headers = {"X-PII-Session-Id": "conversation-7"}
    await client.post(
        "/v1/chat/completions", headers=headers, json={"messages": [{"role": "user", "content": "I'm Grace Hall"}]}
    )
    await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"messages": [{"role": "user", "content": "anna@gmail.com, Grace Hall again"}]},
    )
    assert "<EMAIL_1>, <PERSON_1> again" in upstream_text(recorder)


async def test_tenants_require_a_valid_key(tmp_path: Path, settings: Settings, pipeline: DetectionPipeline) -> None:
    tenants = tmp_path / "tenants.yaml"
    digest = hashlib.sha256(b"tenant-secret").hexdigest()
    tenants.write_text(f"tenants:\n  - {{name: acme, api_key_sha256: {digest}, policy: strict-finance}}\n")
    configured = settings.model_copy(update={"tenants_file": tenants})
    recorder = RecordingProvider(__import__("pii_shield.providers.fake", fromlist=["FakeProvider"]).FakeProvider())
    app = create_app(configured, build_runtime(configured, upstream=recorder, pipeline=pipeline))
    body = {"messages": [{"role": "user", "content": "hello"}]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as http:
        assert (await http.post("/v1/chat/completions", json=body)).status_code == 401
        ok = await http.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer tenant-secret"})
        assert ok.status_code == 200 and ok.headers["x-pii-policy"] == "strict-finance"
        forbidden = await http.post(
            "/p/support-chat/v1/chat/completions", json=body, headers={"Authorization": "Bearer tenant-secret"}
        )
        assert forbidden.status_code == 403
    assert all("tenant-secret" not in json.dumps(r) for r in recorder.requests)  # the client key never goes upstream


async def test_redact_restore_audit_and_health(client: httpx.AsyncClient) -> None:
    redacted = (await client.post("/v1/redact", json={"text": "Card 4111 1111 1111 1111, mail anna@gmail.com"})).json()
    assert redacted["text"] == "Card **** **** **** 1111, mail <EMAIL_1>"
    assert all("original" not in entity for entity in redacted["entities"])
    restored = (
        await client.post("/v1/restore", json={"text": "to <EMAIL_1>", "session_id": redacted["session_id"]})
    ).json()
    assert restored["text"] == "to anna@gmail.com"
    audit = await client.get("/audit")
    assert "anna@gmail.com" not in audit.text and "4111" not in audit.text
    record = audit.json()["records"][0]
    assert record["entity_counts"] == {"CREDIT_CARD": 1, "EMAIL": 1} and len(record["request_hash"]) == 24
    health = (await client.get("/health")).json()
    assert health["status"] == "ok" and "support-chat" in health["policies"]
    assert (await client.get("/v1/models")).json()["data"]


async def test_bad_request_shapes(client: httpx.AsyncClient) -> None:
    assert (await client.post("/v1/chat/completions", json={"messages": []})).status_code == 400


async def test_dashboard_pages(client: httpx.AsyncClient) -> None:
    assert "Playground" in (await client.get("/")).text
    partial = await client.post(
        "/playground",
        data={"text": "I'm Anna Petrova, anna@gmail.com", "policy": "support-chat", "provider": "fake"},
        headers={"HX-Request": "true"},
    )
    assert partial.status_code == 200 and "ent-PERSON" in partial.text and "<!doctype" not in partial.text.lower()
    blocked = await client.post("/playground", data={"text": "card 4111 1111 1111 1111", "policy": "strict-finance"})
    assert "Request refused" in blocked.text
    assert (await client.get("/results")).status_code == 200
    assert "Audit log" in (await client.get("/audit-log")).text


def test_audit_file_failure_does_not_break_requests(tmp_path: Path) -> None:
    from pii_shield.audit import AuditLog
    from pii_shield.shield import RedactionResult

    blocked_dir = tmp_path / "file"
    blocked_dir.write_text("not a directory")
    audit = AuditLog(b"k" * 32, path=blocked_dir / "audit.jsonl")
    record = audit.new_record(
        route="t", result=RedactionResult(text="", session_id="s", policy="p"), request_body="x", tenant=None
    )
    audit.add(record)  # logs an error instead of raising
    assert audit.recent(1)[0].id == record.id
