from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_TOKENS
from .config import ConfigError, Settings
from .embeddings import EmbeddingError, FoundryEmbeddings
from .extract import ExtractionError
from .index import AzureSearchIndex, IndexingError
from .pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Ingest documents into the Azure AI Search RAG index.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data"),
        help="File or directory containing Markdown, text, PDF, or HTML documents.",
    )
    parser.add_argument(
        "--index",
        dest="index_name",
        default=None,
        help="Target index name. Defaults to AZURE_SEARCH_INDEX_NAME or rag-docs.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help=f"Maximum tokens per chunk (default: {DEFAULT_CHUNK_TOKENS}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap tokens between chunks (default: {DEFAULT_CHUNK_OVERLAP}).",
    )
    parser.add_argument(
        "--url-prefix",
        default=None,
        help="Optional URL prefix used to build citation URLs from relative paths.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and chunk documents without calling Azure.",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete and recreate the index before uploading.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if args.chunk_overlap >= args.chunk_tokens:
        parser.error("--chunk-overlap must be smaller than --chunk-tokens")

    try:
        settings = (
            Settings.for_dry_run(args.index_name)
            if args.dry_run
            else Settings.from_env(args.index_name)
        )
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2

    embedder = None
    index = None
    if not args.dry_run:
        embedder = FoundryEmbeddings(settings)
        index = AzureSearchIndex(
            endpoint=settings.search_endpoint,
            index_name=settings.index_name,
        )

    pipeline = IngestionPipeline(
        embedder=embedder,
        index=index,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        upload_batch_size=settings.upload_batch_size,
        codec=None,
    )

    try:
        result = pipeline.run(
            args.source,
            index_name=settings.index_name,
            url_prefix=args.url_prefix,
            dry_run=args.dry_run,
            recreate_index=args.recreate_index,
        )
    except (ExtractionError, EmbeddingError, IndexingError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    mode = "dry-run" if result.dry_run else "uploaded"
    logger.info(
        "Finished (%s): files=%d chunks=%d uploaded=%d index=%s",
        mode,
        result.files,
        result.chunks,
        result.uploaded,
        result.index_name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
