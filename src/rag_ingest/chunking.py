from __future__ import annotations

import re
from typing import Protocol

DEFAULT_CHUNK_TOKENS = 700
DEFAULT_CHUNK_OVERLAP = 100


class TokenCodec(Protocol):
    """Minimal tokenizer interface used by the chunker."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...

    def count(self, text: str) -> int: ...


class TiktokenCodec:
    """Token codec backed by tiktoken's cl100k_base encoding."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    """Normalize line endings and whitespace without destroying paragraphs."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_units(text: str) -> list[str]:
    """Split text into paragraph/sentence units while preserving order."""

    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_BOUNDARY.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                units.append(sentence)
    return units


def _hard_split(unit: str, max_tokens: int, codec: TokenCodec) -> list[str]:
    """Split a unit that is larger than the chunk limit."""

    tokens = codec.encode(unit)
    return [
        codec.decode(tokens[start : start + max_tokens]).strip()
        for start in range(0, len(tokens), max_tokens)
        if tokens[start : start + max_tokens]
    ]


def _tail_units(
    units: list[str], overlap_tokens: int, codec: TokenCodec
) -> tuple[list[str], int]:
    """Return trailing units that fit within the overlap budget."""

    selected: list[str] = []
    total = 0
    for unit in reversed(units):
        unit_tokens = codec.count(unit)
        if total + unit_tokens > overlap_tokens:
            break
        selected.insert(0, unit)
        total += unit_tokens
    return selected, total


def chunk_text(
    text: str,
    *,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
    codec: TokenCodec | None = None,
) -> list[str]:
    """Split text into sentence-aware, token-bounded chunks with overlap."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    normalized = normalize_text(text)
    if not normalized:
        return []

    codec = codec or TiktokenCodec()
    units: list[str] = []
    for unit in _split_units(normalized):
        if codec.count(unit) > max_tokens:
            units.extend(_hard_split(unit, max_tokens, codec))
        else:
            units.append(unit)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = codec.count(unit)

        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append(" ".join(current).strip())
            overlap, overlap_count = _tail_units(current, overlap_tokens, codec)
            if overlap_count + unit_tokens > max_tokens:
                overlap, overlap_count = [], 0
            current = overlap
            current_tokens = overlap_count

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]
