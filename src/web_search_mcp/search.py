from __future__ import annotations

import asyncio
import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote

import httpx
from ddgs import DDGS

from extract import fetch_clean_text
from security import validate_url

logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
    """Raised when web search fails."""


def _normalize_result(item: dict[str, Any]) -> dict[str, str] | None:
    url = item.get("href") or item.get("url")
    if not isinstance(url, str):
        return None
    try:
        validate_url(url)
    except ValueError:
        return None
    return {
        "title": str(item.get("title") or url),
        "url": url,
        "snippet": str(item.get("body") or ""),
    }


# Explicitly exclude the Brave backend. These are free HTML search engines
# scraped by ddgs; Google and Bing are the most reliable fallbacks when
# DuckDuckGo rate-limits cloud egress.
_SEARCH_BACKENDS = ("google", "bing", "duckduckgo")


def _search_sync(query: str, max_results: int) -> list[dict[str, str]]:
    last_error: Exception | None = None
    raw_results: list[dict[str, Any]] = []

    for backend in _SEARCH_BACKENDS:
        for attempt in range(2):
            try:
                with DDGS() as ddgs:
                    raw_results = list(
                        ddgs.text(
                            query,
                            max_results=max_results,
                            backend=backend,
                        )
                    )
                if raw_results:
                    break
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(1.0 + random.uniform(0.0, 0.5))
        if raw_results:
            break

    if not raw_results:
        logger.warning(
            "ddgs returned no results for %r (last error: %s); trying Google News RSS",
            query,
            last_error,
        )
        raw_results = _google_news_rss_sync(query, max_results)

    if not raw_results:
        message = str(last_error) if last_error else "No search results"
        raise SearchError(f"Web search failed: {message}")

    results: list[dict[str, str]] = []
    for item in raw_results:
        if isinstance(item, dict):
            normalized = _normalize_result(item)
            if normalized:
                results.append(normalized)
    return results


def _google_news_rss_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    """Free no-key fallback using Google News RSS."""

    url = (
        "https://news.google.com/rss/search?q="
        f"{quote(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        response = httpx.get(
            url,
            timeout=15,
            headers={"User-Agent": "rag-web-search-mcp/0.1"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except Exception as exc:
        logger.warning("Google News RSS fallback failed: %s", exc)
        return []

    results: list[dict[str, Any]] = []
    for item in root.findall(".//channel/item")[:max_results]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
        if not title or not link:
            continue
        results.append(
            {
                "title": title,
                "href": link,
                "body": " ".join(description.split())[:500],
            }
        )
    return results


async def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    max_results = max(1, min(max_results, 10))
    return await asyncio.to_thread(_search_sync, query, max_results)


async def search_and_read(
    query: str,
    max_results: int = 3,
) -> list[dict[str, str]]:
    results = await search_web(query, max_results=max_results)
    documents: list[dict[str, str]] = []
    for result in results:
        try:
            document = await fetch_clean_text(result["url"])
        except Exception:
            # Google News RSS links often use a JS redirect that trafilatura
            # cannot resolve. Keep the result and use its snippet as context.
            documents.append(
                {
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": result.get("snippet", ""),
                    "text": result.get("snippet", ""),
                }
            )
            continue
        documents.append(
            {
                "title": document["title"],
                "url": document["url"],
                "snippet": result.get("snippet", ""),
                "text": document["text"],
            }
        )
    return documents
