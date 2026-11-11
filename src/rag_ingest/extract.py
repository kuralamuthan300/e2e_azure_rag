from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from .models import SourceDocument

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".html", ".htm"}
_MARKDOWN_HEADING = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


class ExtractionError(RuntimeError):
    """Raised when a source file cannot be converted to text."""


def iter_source_files(source: Path) -> list[Path]:
    """Return supported files from a file or directory, in deterministic order."""

    source = source.expanduser().resolve()
    if not source.exists():
        raise ExtractionError(f"Source does not exist: {source}")

    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ExtractionError(
                f"Unsupported file type {source.suffix!r}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return [source]

    files = [
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(source).parts)
    ]
    if not files:
        raise ExtractionError(f"No supported documents found under {source}")
    return sorted(files)


def _source_root(source: Path) -> Path:
    source = source.expanduser().resolve()
    return source if source.is_dir() else source.parent


def _doc_id(relative_source: str) -> str:
    return hashlib.sha256(relative_source.encode("utf-8")).hexdigest()[:16]


def _title_from_markdown(text: str, fallback: str) -> str:
    match = _MARKDOWN_HEADING.search(text)
    if match:
        return match.group(1).strip()
    return fallback


def _extract_pdf(path: Path) -> tuple[str, dict[str, object]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    title = ""
    if reader.metadata:
        title = str(reader.metadata.get("/Title") or "").strip()
    return "\n\n".join(page.strip() for page in pages if page.strip()), {
        "page_count": len(reader.pages),
        "pdf_title": title,
    }


def _extract_html(path: Path) -> tuple[str, dict[str, object]]:
    import trafilatura

    raw = path.read_text(encoding="utf-8", errors="replace")
    text = trafilatura.extract(
        raw,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    if not text:
        raise ExtractionError(f"Could not extract readable text from {path}")

    title = ""
    try:
        metadata = trafilatura.extract_metadata(raw)
        if metadata and metadata.title:
            title = metadata.title.strip()
    except Exception:
        # Metadata extraction is best-effort; the filename remains the fallback.
        pass
    return text, {"html_title": title}


def _extract_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_document(
    path: Path,
    *,
    root: Path | None = None,
    url_prefix: str | None = None,
) -> SourceDocument:
    """Extract text and metadata from a supported source file."""

    path = path.expanduser().resolve()
    root = (root or _source_root(path)).expanduser().resolve()
    relative_source = path.relative_to(root).as_posix()
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text, extra = _extract_pdf(path)
        title = str(extra.get("pdf_title") or path.stem)
    elif suffix in {".html", ".htm"}:
        text, extra = _extract_html(path)
        title = str(extra.get("html_title") or path.stem)
    elif suffix in {".md", ".markdown"}:
        text = _extract_text(path)
        extra = {}
        title = _title_from_markdown(text, path.stem)
    elif suffix == ".txt":
        text = _extract_text(path)
        extra = {}
        title = path.stem
    else:
        raise ExtractionError(
            f"Unsupported file type {suffix!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if not text.strip():
        raise ExtractionError(f"No readable text found in {path}")

    stat = path.stat()
    metadata: dict[str, object] = {
        "extension": suffix,
        "relative_path": relative_source,
        "size_bytes": stat.st_size,
        **extra,
    }
    url = None
    if url_prefix:
        url = f"{url_prefix.rstrip('/')}/{quote(relative_source)}"

    return SourceDocument(
        doc_id=_doc_id(relative_source),
        title=title or path.stem,
        source=relative_source,
        text=text,
        last_updated=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        url=url,
        metadata=metadata,
    )
