from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from models import RetrievedDocument, WebDocument

_AZURE_AI_SCOPE = "https://ai.azure.com/.default"
_FOUNDRY_FEATURES_HEADER = "Foundry-Features"
_FOUNDRY_FEATURES = "Toolboxes=V1Preview"
_FOUNDRY_CALL_ID_HEADER = "x-agent-foundry-call-id"
logger = logging.getLogger(__name__)


def _flatten_exception(exc: BaseException) -> list[str]:
    nested = getattr(exc, "exceptions", None)
    if nested:
        messages: list[str] = []
        for child in nested:
            messages.extend(_flatten_exception(child))
        return messages
    return [str(exc)]


def _platform_call_id() -> str | None:
    """Read the per-request Foundry call ID when running in Agent Server."""

    try:
        from azure.ai.agentserver.core import get_request_context
    except ImportError:
        return None

    try:
        return get_request_context().platform_headers().get(_FOUNDRY_CALL_ID_HEADER)
    except Exception:
        return None


class ToolboxError(RuntimeError):
    """Raised when a Foundry Toolbox MCP call fails."""


@dataclass(slots=True)
class ToolboxClient:
    """Minimal MCP client for the Foundry Toolbox endpoint."""

    project_endpoint: str
    toolbox_name: str
    api_version: str = "v1"
    timeout_seconds: float = 30.0
    credential: TokenCredential = field(default_factory=DefaultAzureCredential)

    @property
    def endpoint(self) -> str:
        project = self.project_endpoint.rstrip("/")
        return (
            f"{project}/toolboxes/{self.toolbox_name}"
            f"/mcp?api-version={self.api_version}"
        )

    async def _headers(self) -> dict[str, str]:
        token = await asyncio.to_thread(
            self.credential.get_token,
            _AZURE_AI_SCOPE,
        )
        headers = {
            "Authorization": f"Bearer {token.token}",
            _FOUNDRY_FEATURES_HEADER: _FOUNDRY_FEATURES,
        }
        call_id = _platform_call_id()
        if call_id:
            headers["x-agent-foundry-call-id"] = call_id
        logger.info("Toolbox request platform_call_id=%s", bool(call_id))
        return headers

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        headers = await self._headers()
        try:
            async with (
                streamablehttp_client(
                    self.endpoint,
                    headers=headers,
                    timeout=self.timeout_seconds,
                ) as (read_stream, write_stream, _),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                result = await session.call_tool(name, arguments)
        except BaseExceptionGroup as exc:
            details = "; ".join(_flatten_exception(exc))
            raise ToolboxError(
                f"Toolbox call {name!r} failed: {details}"
            ) from exc
        except Exception as exc:
            raise ToolboxError(f"Toolbox call {name!r} failed: {exc}") from exc

        if result.isError:
            message = "\n".join(
                getattr(item, "text", str(item)) for item in result.content
            )
            raise ToolboxError(f"Toolbox tool {name!r} returned an error: {message}")

        if result.structuredContent:
            return dict(result.structuredContent)

        # Some MCP servers return JSON inside a text content block.
        for item in result.content:
            text = getattr(item, "text", None)
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        raise ToolboxError(f"Toolbox tool {name!r} returned no structured content")

    async def search(self, query: str, *, top_k: int = 5) -> list[RetrievedDocument]:
        payload = await self.call_tool("search", {"query": query})
        raw_documents = payload.get("documents")
        if not isinstance(raw_documents, list):
            return []
        return [
            RetrievedDocument.from_tool_document(document)
            for document in raw_documents[:top_k]
            if isinstance(document, dict)
        ]

    async def web_search(
        self,
        query: str,
        *,
        max_results: int = 3,
    ) -> list[WebDocument]:
        tool_names = await self.list_tools()
        candidates = [
            name
            for name in tool_names
            if name.endswith(("search_and_read", "search_and_read_tool"))
        ]
        if not candidates:
            candidates = [
                name
                for name in tool_names
                if name.endswith(("search_web", "search_web_tool"))
            ]
        if not candidates:
            raise ToolboxError("No web search tool is attached to the toolbox")

        payload = await self.call_tool(
            candidates[0],
            {"query": query, "max_results": max_results},
        )
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raw_results = payload.get("documents")
        if not isinstance(raw_results, list):
            return []
        return [
            WebDocument.from_tool_document(document)
            for document in raw_results[:max_results]
            if isinstance(document, dict)
        ]

    async def list_tools(self) -> list[str]:
        headers = await self._headers()
        try:
            async with (
                streamablehttp_client(
                    self.endpoint,
                    headers=headers,
                    timeout=self.timeout_seconds,
                ) as (read_stream, write_stream, _),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                result = await session.list_tools()
        except Exception as exc:
            raise ToolboxError(f"Could not list toolbox tools: {exc}") from exc
        return [tool.name for tool in result.tools]
