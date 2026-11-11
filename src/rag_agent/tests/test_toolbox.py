from __future__ import annotations

import pytest

from models import RetrievedDocument
from toolbox import ToolboxClient, ToolboxError


def make_client() -> ToolboxClient:
    return ToolboxClient(
        project_endpoint="https://example.services.ai.azure.com/api/projects/demo",
        toolbox_name="rag-tools",
    )


async def test_web_search_prefers_search_and_read_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client()
    calls: list[tuple[str, dict]] = []

    async def fake_list_tools(self: ToolboxClient) -> list[str]:
        return ["search", "websearch___search_and_read_tool", "websearch___search_web_tool"]

    async def fake_call_tool(
        self: ToolboxClient,
        name: str,
        arguments: dict,
    ) -> dict:
        calls.append((name, arguments))
        return {
            "results": [
                {
                    "title": "Refund guidance",
                    "url": "https://example.com/refunds",
                    "snippet": "Refunds take 30 days.",
                    "text": "Refunds take 30 days.",
                }
            ]
        }

    monkeypatch.setattr(ToolboxClient, "list_tools", fake_list_tools)
    monkeypatch.setattr(ToolboxClient, "call_tool", fake_call_tool)

    results = await client.web_search("refund policy", max_results=1)

    assert calls[0][0] == "websearch___search_and_read_tool"
    assert calls[0][1] == {"query": "refund policy", "max_results": 1}
    assert results[0].url == "https://example.com/refunds"


async def test_web_search_falls_back_to_search_web_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client()

    async def fake_list_tools(self: ToolboxClient) -> list[str]:
        return ["websearch___search_web_tool"]

    async def fake_call_tool(
        self: ToolboxClient,
        name: str,
        arguments: dict,
    ) -> dict:
        return {"results": [{"title": "T", "url": "https://example.com", "snippet": "S"}]}

    monkeypatch.setattr(ToolboxClient, "list_tools", fake_list_tools)
    monkeypatch.setattr(ToolboxClient, "call_tool", fake_call_tool)

    results = await client.web_search("query")

    assert results[0].url == "https://example.com"


async def test_web_search_without_web_tool_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client()

    async def fake_list_tools(self: ToolboxClient) -> list[str]:
        return ["search"]

    monkeypatch.setattr(ToolboxClient, "list_tools", fake_list_tools)

    with pytest.raises(ToolboxError, match="No web search tool"):
        await client.web_search("query")


async def test_search_maps_tool_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client()

    async def fake_call_tool(
        self: ToolboxClient,
        name: str,
        arguments: dict,
    ) -> dict:
        assert name == "search"
        return {
            "documents": [
                {
                    "id": "chunk-1",
                    "title": "Refund policy",
                    "content": "Refunds take 30 days.",
                    "url": "https://example.com/refunds",
                    "filepath": "refunds.md",
                    "score": 0.82,
                },
                "not-a-document",
            ]
        }

    monkeypatch.setattr(ToolboxClient, "call_tool", fake_call_tool)

    documents = await client.search("refund policy", top_k=5)

    assert len(documents) == 1
    assert documents[0].id == "chunk-1"
    assert documents[0].source == "refunds.md"
    assert documents[0].score == pytest.approx(0.82)


def test_document_defaults_for_missing_fields() -> None:
    document = RetrievedDocument.from_tool_document({"content": "text"})

    assert document.id == ""
    assert document.title == "Untitled document"
    assert document.url is None
    assert document.source is None
    assert document.score is None
