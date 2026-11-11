from __future__ import annotations

import pytest

from rag_ingest.chunking import chunk_text, normalize_text
from tests.conftest import WordCodec


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_text("   \n\n ", codec=WordCodec()) == []


def test_chunks_respect_max_tokens() -> None:
    text = " ".join(f"Sentence number {index} has several words." for index in range(40))
    codec = WordCodec()

    chunks = chunk_text(text, max_tokens=12, overlap_tokens=4, codec=codec)

    assert len(chunks) > 1
    assert all(codec.count(chunk) <= 12 for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_overlap_repeats_trailing_sentence() -> None:
    text = (
        "Alpha beta gamma delta. "
        "Epsilon zeta eta theta. "
        "Iota kappa lambda mu. "
        "Nu xi omicron pi. "
    )
    chunks = chunk_text(text, max_tokens=8, overlap_tokens=4, codec=WordCodec())

    assert len(chunks) >= 2
    assert chunks[1].startswith("Epsilon zeta eta theta.")


def test_oversized_sentence_is_hard_split() -> None:
    text = " ".join(f"token{index}" for index in range(30))
    codec = WordCodec()

    chunks = chunk_text(text, max_tokens=7, overlap_tokens=2, codec=codec)

    assert len(chunks) > 1
    assert all(codec.count(chunk) <= 7 for chunk in chunks)


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_text("hello world", max_tokens=10, overlap_tokens=10, codec=WordCodec())


def test_normalize_text_collapses_extra_whitespace() -> None:
    assert normalize_text(" hello\t world \r\n\r\n\r\nnext ") == "hello world\n\nnext"
