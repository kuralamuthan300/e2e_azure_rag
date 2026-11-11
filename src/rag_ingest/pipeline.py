from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from .chunking import TokenCodec, chunk_text
from .embeddings import EmbeddingProvider
from .extract import extract_document, iter_source_files
from .index import AzureSearchIndex
from .models import Chunk, IngestionResult, SourceDocument

logger = logging.getLogger(__name__)


def document_to_chunks(
    document: SourceDocument,
    *,
    chunk_tokens: int,
    chunk_overlap: int,
    codec: TokenCodec | None = None,
) -> list[Chunk]:
    """Convert an extracted document into search-ready chunks."""

    pieces = chunk_text(
        document.text,
        max_tokens=chunk_tokens,
        overlap_tokens=chunk_overlap,
        codec=codec,
    )
    chunks: list[Chunk] = []
    for index, content in enumerate(pieces):
        chunk_id = f"{index:04d}"
        metadata = {
            **document.metadata,
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
        }
        chunks.append(
            Chunk(
                id=f"{document.doc_id}-{chunk_id}",
                doc_id=document.doc_id,
                chunk_id=chunk_id,
                title=document.title,
                content=content,
                source=document.source,
                last_updated=document.last_updated,
                chunk_index=index,
                url=document.url,
                metadata=metadata,
            )
        )
    return chunks


class IngestionPipeline:
    """Extract, chunk, embed, and upload documents to Azure AI Search."""

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider | None,
        index: AzureSearchIndex | None,
        chunk_tokens: int = 700,
        chunk_overlap: int = 100,
        upload_batch_size: int = 100,
        codec: TokenCodec | None = None,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._chunk_tokens = chunk_tokens
        self._chunk_overlap = chunk_overlap
        self._upload_batch_size = upload_batch_size
        self._codec = codec

    def run(
        self,
        source: Path,
        *,
        index_name: str,
        url_prefix: str | None = None,
        dry_run: bool = False,
        recreate_index: bool = False,
    ) -> IngestionResult:
        source = source.expanduser().resolve()
        files = iter_source_files(source)
        root = source if source.is_dir() else source.parent
        result = IngestionResult(
            source=str(source),
            index_name=index_name,
            files=len(files),
            dry_run=dry_run,
        )

        embedder = self._embedder
        index = self._index
        if not dry_run:
            if embedder is None or index is None:
                raise RuntimeError("Embedder and index are required for a real run")
            index.ensure_index(
                embedder.dimensions,
                recreate=recreate_index,
            )

        for path in files:
            document = extract_document(path, root=root, url_prefix=url_prefix)
            chunks = document_to_chunks(
                document,
                chunk_tokens=self._chunk_tokens,
                chunk_overlap=self._chunk_overlap,
                codec=self._codec,
            )
            if not chunks:
                logger.warning("No chunks produced for %s", document.source)
                continue

            result.chunks += len(chunks)
            if dry_run:
                logger.info(
                    "[dry-run] %s -> %d chunk(s)", document.source, len(chunks)
                )
                continue

            if embedder is None or index is None:
                raise RuntimeError("Embedder and index are required for a real run")
            vectors = embedder.embed([chunk.content for chunk in chunks])
            documents = [
                chunk.to_search_document(vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            uploaded = index.upload(
                documents,
                batch_size=self._upload_batch_size,
            )
            result.uploaded += uploaded
            logger.info(
                "%s -> %d chunk(s), %d uploaded", document.source, len(chunks), uploaded
            )

        return result
