from __future__ import annotations

import pytest

from security import UnsafeUrlError, validate_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://169.254.169.254/metadata",
        "http://10.0.0.1/",
        "file:///etc/passwd",
        "ftp://example.com/file",
        "https://user:pass@example.com/",
    ],
)
def test_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(UnsafeUrlError):
        validate_url(url, resolve_dns=False)


def test_accepts_public_https_url_without_dns() -> None:
    assert validate_url("https://example.com/article", resolve_dns=False) == (
        "https://example.com/article"
    )
