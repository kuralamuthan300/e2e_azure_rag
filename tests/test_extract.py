from __future__ import annotations

from pathlib import Path

import pytest

from rag_ingest.extract import ExtractionError, extract_document, iter_source_files


def test_extract_markdown_title_and_url(tmp_path: Path) -> None:
    document_path = tmp_path / "policies" / "refunds.md"
    document_path.parent.mkdir()
    document_path.write_text("# Refund Policy\n\nRefunds are available within 30 days.\n")

    document = extract_document(
        document_path,
        root=tmp_path,
        url_prefix="https://docs.example.com/",
    )

    assert document.title == "Refund Policy"
    assert "30 days" in document.text
    assert document.source == "policies/refunds.md"
    assert document.url == "https://docs.example.com/policies/refunds.md"
    assert document.metadata["relative_path"] == "policies/refunds.md"


def test_doc_id_is_deterministic(tmp_path: Path) -> None:
    document_path = tmp_path / "guide.txt"
    document_path.write_text("Some guide text.")

    first = extract_document(document_path, root=tmp_path)
    second = extract_document(document_path, root=tmp_path)

    assert first.doc_id == second.doc_id
    assert first.title == "guide"


def test_unsupported_file_rejected(tmp_path: Path) -> None:
    document_path = tmp_path / "notes.docx"
    document_path.write_text("not really a docx")

    with pytest.raises(ExtractionError, match="Unsupported file type"):
        iter_source_files(document_path)


def test_empty_document_rejected(tmp_path: Path) -> None:
    document_path = tmp_path / "empty.md"
    document_path.write_text("   \n")

    with pytest.raises(ExtractionError, match="No readable text"):
        extract_document(document_path, root=tmp_path)
