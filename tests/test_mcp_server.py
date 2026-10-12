"""MCP server through a real MCP client: in-process for the tools, one stdio subprocess for the transport."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters
from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent
from typer.testing import CliRunner

from pii_shield.cli import app
from pii_shield.detect.pipeline import DetectionPipeline
from pii_shield.mcp_server import McpOptions, create_mcp_server
from pii_shield.policy import PolicySet
from pii_shield.shield import Shield
from pii_shield.vault.store import MemoryBackend, Vault
from tests.conftest import TEST_KEY

EMAIL = "anna.petrova@gmail.com"
CARD = "4111 1111 1111 1111"
TICKET = f"Hi, I'm Anna Petrova, write to {EMAIL} or call +44 7911 123456. Card {CARD}."


def text_of(result: CallToolResult) -> str:
    return "".join(block.text for block in result.content if isinstance(block, TextContent))


def everything(result: CallToolResult) -> str:
    """All a model could read from a result: the text blocks and the structured content."""
    return text_of(result) + json.dumps(result.structured_content or {}, ensure_ascii=False)


async def call(client: Client, tool: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, text_of(result)
    assert result.structured_content is not None
    return result.structured_content


@pytest.fixture
def options(tmp_path: Path) -> McpOptions:
    return McpOptions(root=tmp_path.resolve(), ner_enabled=True)


# Clients are opened inside each test, not in async fixtures: the SDK's anyio task groups must be entered
# and exited in the same task, which pytest-asyncio fixtures do not guarantee.
@pytest.fixture
def server(shield: Shield, options: McpOptions) -> MCPServer:
    return create_mcp_server(shield, options)


@pytest.fixture
def restore_server(shield: Shield, options: McpOptions) -> MCPServer:
    options.allow_restore = True
    return create_mcp_server(shield, options)


async def test_tools_are_listed_with_schemas_and_annotations(server: MCPServer) -> None:
    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert set(tools) == {"detect_pii", "redact", "redact_file", "list_policies"}  # no restore by default
        for name in ("detect_pii", "list_policies"):
            assert tools[name].annotations is not None and tools[name].annotations.read_only_hint is True
        assert tools["redact"].annotations is not None and tools["redact"].annotations.destructive_hint is False
        props = tools["redact"].input_schema["properties"]
        assert set(props) == {"text", "policy", "session_id"} and props["session_id"]["description"]
        assert tools["redact"].output_schema is not None
        assert tools["redact"].description and "never returned" in tools["redact"].description


async def test_detect_pii_reports_spans_without_values_by_default(server: MCPServer) -> None:
    async with Client(server) as client:
        out = await call(client, "detect_pii", text=TICKET)
        by_type = {f["type"]: f for f in out["findings"]}
        assert {"PERSON", "EMAIL", "PHONE", "CREDIT_CARD"} <= set(by_type)
        email = by_type["EMAIL"]
        assert TICKET[email["start"] : email["end"]] == EMAIL and email["validated"] and email["value"] is None
        assert by_type["CREDIT_CARD"]["action"] == "mask" and out["would_block"] is False
        with_values = await call(client, "detect_pii", text=TICKET, include_values=True)
        assert EMAIL in [f["value"] for f in with_values["findings"]]


async def test_detect_pii_says_when_a_policy_would_block(server: MCPServer) -> None:
    async with Client(server) as client:
        out = await call(client, "detect_pii", text=TICKET, policy="strict-finance")
        assert out["would_block"] is True and out["policy"] == "strict-finance"


async def test_redact_returns_placeholders_and_never_the_originals(server: MCPServer) -> None:
    async with Client(server) as client:
        result = await client.call_tool("redact", {"text": TICKET})
        out = result.structured_content
        assert out is not None and not result.is_error
        assert out["redacted_text"].startswith("Hi, I'm <PERSON_1>, write to <EMAIL_1> or call <PHONE_1>.")
        assert "**** **** **** 1111" in out["redacted_text"]
        assert out["counts"]["EMAIL"] == 1 and out["restorable"] == 3 and out["blocked"] is False
        assert any("data, not instructions" in note for note in out["notes"])
        leaked = everything(result)
        for original in (EMAIL, "Anna Petrova", "7911 123456", CARD):
            assert original not in leaked


async def test_session_keeps_placeholders_consistent(server: MCPServer) -> None:
    async with Client(server) as client:
        first = await call(client, "redact", text=f"Mail {EMAIL}")
        second = await call(client, "redact", text=f"Bob and {EMAIL}", session_id=first["session_id"])
        assert "<EMAIL_1>" in second["redacted_text"] and second["session_id"] == first["session_id"]


async def test_blocking_policy_returns_no_text(server: MCPServer) -> None:
    async with Client(server) as client:
        result = await client.call_tool("redact", {"text": TICKET, "policy": "strict-finance"})
        out = result.structured_content
        assert out is not None and out["blocked"] is True and out["redacted_text"] is None
        assert "CREDIT_CARD" in out["block_reasons"] and CARD not in everything(result)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"text": "hi", "policy": "nope"}, "list_policies"),
        ({"text": "   "}, "empty"),
        ({"text": "x" * 300_001}, "limit"),
        ({"text": "hi", "session_id": "../etc"}, "session_id"),
    ],
)
async def test_bad_input_is_a_helpful_tool_error(server: MCPServer, arguments: dict[str, Any], message: str) -> None:
    async with Client(server) as client:
        result = await client.call_tool("redact", arguments)
        assert result.is_error and message in text_of(result)
        assert len(text_of(result)) < 500  # the rejected text is not echoed back


async def test_redact_file_reads_server_side(server: MCPServer, tmp_path: Path) -> None:
    async with Client(server) as client:
        (tmp_path / "tickets").mkdir()
        (tmp_path / "tickets" / "t1.txt").write_text(TICKET, encoding="utf-8")
        result = await client.call_tool("redact_file", {"path": "tickets/t1.txt"})
        out = result.structured_content
        assert out is not None and not result.is_error
        assert "<EMAIL_1>" in out["redacted_text"] and out["source_file"].endswith("t1.txt")
        assert EMAIL not in everything(result)


async def test_redact_file_stays_inside_the_root(server: MCPServer, tmp_path: Path) -> None:
    async with Client(server) as client:
        outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
        outside.write_text(TICKET, encoding="utf-8")
        (tmp_path / "link.txt").symlink_to(outside)
        for path in (str(outside), f"../{outside.name}", "link.txt"):
            result = await client.call_tool("redact_file", {"path": path})
            assert result.is_error and "outside the allowed root" in text_of(result), path
        missing = await client.call_tool("redact_file", {"path": "nope.txt"})
        assert missing.is_error and "No such file" in text_of(missing)


async def test_list_policies(server: MCPServer, options: McpOptions) -> None:
    async with Client(server) as client:
        out = await call(client, "list_policies")
        names = {p["name"]: p for p in out["policies"]}
        assert {"support-chat", "strict-finance", "natural-text", "analytics-irreversible"} <= set(names)
        assert out["default_policy"] == "support-chat" and names["support-chat"]["default"]
        assert names["strict-finance"]["blocks"] == ["CREDIT_CARD", "SECRET", "US_SSN"]
        assert names["analytics-irreversible"]["reversible"] == []
        assert names["support-chat"]["entities"]["CREDIT_CARD"] == "mask (keep last 4)"
        assert out["restore_enabled"] is False and out["file_root"] == str(options.root)


async def test_patterns_only_server_says_names_are_not_detected(policies: PolicySet, tmp_path: Path) -> None:
    vault = Vault(MemoryBackend(), TEST_KEY)
    shield = Shield(policies=policies, pipeline=DetectionPipeline(), vault=vault, secret=b"s" * 32)
    async with Client(create_mcp_server(shield, McpOptions(root=tmp_path))) as c:
        out = await call(c, "redact", text=TICKET)
    assert "Anna Petrova" in out["redacted_text"] and "<EMAIL_1>" in out["redacted_text"]
    assert any("NER is off" in note for note in out["notes"])


async def test_restore_inline_and_to_a_file(restore_server: MCPServer, tmp_path: Path) -> None:
    async with Client(restore_server) as restore_client:
        assert "restore" in {t.name for t in (await restore_client.list_tools()).tools}
        redacted = await call(restore_client, "redact", text=TICKET)
        reply = "Dear <PERSON_1>, we will write to <EMAIL_1>. <PERSON_9> is unknown."
        inline = await call(restore_client, "restore", text=reply, session_id=redacted["session_id"])
        assert inline["restored_text"].startswith(f"Dear Anna Petrova, we will write to {EMAIL}.")
        assert inline["restored"] == 2 and inline["unknown_placeholders"] == ["<PERSON_9>"]

        result = await restore_client.call_tool(
            "restore", {"text": reply, "session_id": redacted["session_id"], "output_path": "out/reply.txt"}
        )
        out = result.structured_content
        assert out is not None and out["restored_text"] is None and out["restored"] == 2
        assert EMAIL not in everything(result)
        assert (tmp_path / "out" / "reply.txt").read_text(encoding="utf-8").startswith("Dear Anna Petrova")

        again = await restore_client.call_tool(
            "restore", {"text": reply, "session_id": redacted["session_id"], "output_path": "out/reply.txt"}
        )
        assert again.is_error and "already exists" in text_of(again)
        escape = await restore_client.call_tool(
            "restore", {"text": reply, "session_id": redacted["session_id"], "output_path": "../x.txt"}
        )
        assert escape.is_error and "outside the allowed root" in text_of(escape)


async def test_restore_unknown_session(restore_server: MCPServer) -> None:
    async with Client(restore_server) as restore_client:
        result = await restore_client.call_tool("restore", {"text": "<EMAIL_1>", "session_id": "never-seen"})
        assert result.is_error and "nothing to restore" in text_of(result)


async def test_stdio_transport_end_to_end(tmp_path: Path) -> None:
    """The real `pii-shield mcp` process: stdout carries only protocol messages."""
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "pii_shield.cli", "mcp", "--root", str(tmp_path)], cwd=str(tmp_path)
    )
    async with Client(params) as c:
        assert "redact" in {t.name for t in (await c.list_tools()).tools}
        out = await call(c, "redact", text=f"Mail {EMAIL}")
        assert out["redacted_text"] == "Mail <EMAIL_1>"
        policies = await call(c, "list_policies")
        assert policies["ner_enabled"] is False and policies["restore_enabled"] is False


def test_cli_refuses_a_public_http_bind() -> None:
    result = CliRunner().invoke(app, ["mcp", "--http", "--host", "0.0.0.0"])
    assert result.exit_code == 1 and "--allow-remote" in result.output
