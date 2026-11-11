from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

import server


async def test_tools_list_exposes_contract() -> None:
    async with create_connected_server_and_client_session(server.mcp) as client:
        result = await client.list_tools()

    tools = {tool.name: tool for tool in result.tools}

    assert {
        "search_web_tool",
        "fetch_clean_text_tool",
        "search_and_read_tool",
    } <= tools.keys()

    search_and_read = tools["search_and_read_tool"]
    assert search_and_read.inputSchema["properties"]["query"]["type"] == "string"
    assert "max_results" in search_and_read.inputSchema["properties"]

    fetch = tools["fetch_clean_text_tool"]
    assert fetch.inputSchema["properties"]["url"]["type"] == "string"
    assert "url" in fetch.inputSchema.get("required", [])


async def test_search_web_tool_returns_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search_web(query: str, max_results: int = 5) -> list[dict]:
        assert query == "refund policy"
        assert max_results == 2
        return [
            {
                "title": "Refund policy",
                "url": "https://example.com/refunds",
                "snippet": "Refunds take 30 days.",
            }
        ]

    monkeypatch.setattr(server, "search_web", fake_search_web)

    async with create_connected_server_and_client_session(server.mcp) as client:
        result = await client.call_tool(
            "search_web_tool",
            {"query": "refund policy", "max_results": 2},
        )

    payload = result.structuredContent or {}
    if not payload:
        payload = {
            "results": [
                item
                for content in result.content
                for item in (getattr(content, "text", None),)
                if item
            ]
        }

    assert payload["results"][0]["url"] == "https://example.com/refunds"


async def test_search_web_tool_hides_search_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing_search_web(query: str, max_results: int = 5) -> list[dict]:
        raise server.SearchError("search backend unavailable")

    monkeypatch.setattr(server, "search_web", failing_search_web)

    async with create_connected_server_and_client_session(server.mcp) as client:
        result = await client.call_tool("search_web_tool", {"query": "refund policy"})

    payload = result.structuredContent or {}
    assert payload.get("results") == []
    assert "search backend unavailable" in str(payload.get("error", ""))
