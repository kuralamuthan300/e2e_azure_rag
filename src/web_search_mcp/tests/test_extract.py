from __future__ import annotations

import pytest

from extract import fetch_clean_text
from security import UnsafeUrlError


@pytest.mark.asyncio
async def test_fetch_rejects_private_url() -> None:
    with pytest.raises(UnsafeUrlError):
        await fetch_clean_text("http://127.0.0.1/")
