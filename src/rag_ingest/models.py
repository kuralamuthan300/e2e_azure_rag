from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A source file after text extraction."""

    doc_id: str
    title: str
    source: str
    text: str
    last_updated: datetime
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A token-bounded chunk ready for embedding and upload."""

    id: str
    doc_id: str
    chunk_id: str
    title: str
    content: str
    source: str
    last_updated: datetime
    chunk_index: int
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_search_document(self, vector: list[float]) -> dict[str, Any]:
        """Convert the chunk to the JSON shape stored in Azure AI Search."""

        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "content": self.content,
            "content_vector": vector,
            "source": self.source,
            "url": self.url,
            "chunk_index": self.chunk_index,
            "last_updated": self.last_updated.isoformat(),
            "metadata": json.dumps(self.metadata, sort_keys=True),
        }


@dataclass(slots=True)
class IngestionResult:
    """Summary returned by an ingestion run."""

    source: str
    index_name: str
    files: int = 0
    chunks: int = 0
    uploaded: int = 0
    dry_run: bool = False
