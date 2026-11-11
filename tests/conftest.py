from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WordCodec:
    """Deterministic token codec for tests; one token per whitespace word."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, tokens: list[int]) -> str:
        return " ".join(f"w{token}" for token in tokens)

    def count(self, text: str) -> int:
        return len(text.split())
