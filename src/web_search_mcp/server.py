from __future__ import annotations

import os
import secrets
from collections.abc import Awaitable, Callable
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from extract import FetchError, fetch_clean_text
from search import SearchError, search_and_read, search_web
from security import UnsafeUrlError

_DEFAULT_HOSTS = "localhost,127.0.0.1,localhost:8000,127.0.0.1:8000"


def _transport_security() -> TransportSecuritySettings:
    hosts = [
        item.strip()
        for item in os.getenv("WEB_SEARCH_MCP_ALLOWED_HOSTS", _DEFAULT_HOSTS).split(",")
        if item.strip()
    ]
    origins = [
        f"{scheme}://{host}"
        for host in hosts
        for scheme in ("http", "https")
    ]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


mcp = FastMCP(
    "web-search-mcp",
    instructions=(
        "Search the public web and return clean, citation-ready page text. "
        "All results include a URL."
    ),
    stateless_http=True,
    streamable_http_path="/mcp",
    transport_security=_transport_security(),
)


@mcp.tool()
async def search_web_tool(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search DuckDuckGo and return result titles, URLs, and snippets."""

    try:
        results = await search_web(query, max_results=max_results)
    except SearchError as exc:
        return {"error": str(exc), "results": []}
    return {"results": results}


@mcp.tool()
async def fetch_clean_text_tool(url: str) -> dict[str, Any]:
    """Fetch a public web page and return trafilatura-extracted text."""

    try:
        return await fetch_clean_text(url)
    except (FetchError, UnsafeUrlError) as exc:
        return {"error": str(exc), "url": url, "text": ""}


@mcp.tool()
async def search_and_read_tool(query: str, max_results: int = 3) -> dict[str, Any]:
    """Search the web and return extracted text for the top results."""

    try:
        results = await search_and_read(query, max_results=max_results)
    except SearchError as exc:
        return {"error": str(exc), "results": []}
    return {"results": results}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, token: str) -> None:
        super().__init__(app)
        self._expected = f"Bearer {token}"

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in {"/healthz", "/readyz"}:
            return await call_next(request)
        provided = request.headers.get("authorization", "")
        if not secrets.compare_digest(provided, self._expected):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


async def healthz(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def create_app() -> Any:
    token = os.getenv("WEB_SEARCH_MCP_TOKEN", "").strip()
    if not token:
        raise RuntimeError("WEB_SEARCH_MCP_TOKEN is required")

    app = mcp.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware, token=token)
    app.routes.insert(0, Route("/healthz", healthz, methods=["GET"]))
    app.routes.insert(0, Route("/readyz", healthz, methods=["GET"]))
    return app


def main() -> None:
    uvicorn.run(
        create_app(),
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
