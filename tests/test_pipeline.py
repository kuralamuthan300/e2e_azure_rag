from __future__ import annotations

import json
from pathlib import Path

from rag_ingest.extract import extract_document
from rag_ingest.pipeline import IngestionPipeline, document_to_chunks
from tests.conftest import WordCodec


class FakeEmbedder:
    dimensions = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeIndex:
    def __init__(self) -> None:
        self.ensured: list[tuple[int, bool]] = []
        self.uploaded: list[dict[str, object]] = []

    def ensure_index(self, dimensions: int, *, recreate: bool = False) -> None:
        self.ensured.append((dimensions, recreate))

    def upload(self, documents: list[dict[str, object]], *, batch_size: int = 100) -> int:
        self.uploaded.extend(documents)
        return len(documents)


def test_pipeline_uploads_embeddings_and_metadata(tmp_path: Path) -> None:
    document_path = tmp_path / "refunds.md"
    document_path.write_text("# Refund Policy\n\nRefunds are available within 30 days.")

    embedder = FakeEmbedder()
    index = FakeIndex()
    pipeline = IngestionPipeline(
        embedder=embedder,
        index=index,
        chunk_tokens=20,
        chunk_overlap=4,
        codec=WordCodec(),
    )

    result = pipeline.run(
        tmp_path,
        index_name="rag-docs",
        url_prefix="https://docs.example.com",
    )

    assert result.files == 1
    assert result.chunks == 1
    assert result.uploaded == 1
    assert index.ensured == [(3, False)]
    assert embedder.calls == [["# Refund Policy Refunds are available within 30 days."]]
    assert index.uploaded[0]["content_vector"] == [0.1, 0.2, 0.3]
    assert index.uploaded[0]["source"] == "refunds.md"
    assert json.loads(index.uploaded[0]["metadata"])["relative_path"] == "refunds.md"


def test_pipeline_dry_run_does_not_touch_azure(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("# Guide\n\nUseful content.")

    pipeline = IngestionPipeline(
        embedder=None,
        index=None,
        codec=WordCodec(),
    )

    result = pipeline.run(tmp_path, index_name="rag-docs", dry_run=True)

    assert result.files == 1
    assert result.chunks == 1
    assert result.uploaded == 0
    assert result.dry_run is True


def test_chunk_ids_are_deterministic(tmp_path: Path) -> None:
    document_path = tmp_path / "guide.md"
    document_path.write_text("# Guide\n\nUseful content. More useful content.")

    document = extract_document(document_path, root=tmp_path)
    first = document_to_chunks(
        document,
        chunk_tokens=3,
        chunk_overlap=1,
        codec=WordCodec(),
    )
    second = document_to_chunks(
        document,
        chunk_tokens=3,
        chunk_overlap=1,
        codec=WordCodec(),
    )

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert len({chunk.id for chunk in first}) == len(first)
