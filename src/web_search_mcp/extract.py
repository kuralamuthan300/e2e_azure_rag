from __future__ import annotations

from urllib.parse import urljoin

import httpx
import trafilatura

from security import validate_url

MAX_RESPONSE_BYTES = 2_000_000
MAX_TEXT_CHARS = 20_000
USER_AGENT = "rag-web-search-mcp/0.1 (+https://github.com/)"


class FetchError(RuntimeError):
    """Raised when a page cannot be fetched or extracted."""


async def fetch_clean_text(url: str, *, max_chars: int = MAX_TEXT_CHARS) -> dict[str, str]:
    """Fetch a page, follow validated redirects, and extract readable text."""

    current = validate_url(url)
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain"}

    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(15.0, connect=10.0),
        headers=headers,
    ) as client:
        for _ in range(4):
            try:
                response = await client.get(current)
            except httpx.HTTPError as exc:
                raise FetchError(f"Request failed for {current}: {exc}") from exc

            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise FetchError(f"Redirect without Location from {current}")
                current = validate_url(urljoin(current, location))
                continue

            response.raise_for_status()
            if len(response.content) > MAX_RESPONSE_BYTES:
                raise FetchError(f"Response too large for {current}")
            break
        else:
            raise FetchError(f"Too many redirects for {url}")

    content_type = response.headers.get("content-type", "").lower()
    raw = response.content.decode(response.encoding or "utf-8", errors="replace")

    if "text/plain" in content_type:
        text = raw
        title = current
    else:
        text = trafilatura.extract(
            raw,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
        title = current
        try:
            metadata = trafilatura.extract_metadata(raw)
            if metadata and metadata.title:
                title = metadata.title
        except Exception:
            pass

    if not text or not text.strip():
        raise FetchError(f"No readable text found at {current}")

    return {
        "url": current,
        "title": title or current,
        "text": text.strip()[:max_chars],
    }
