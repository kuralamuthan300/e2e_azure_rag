from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rag_agent"))

from langchain_core.messages import HumanMessage  # noqa: E402

from main import create_graph  # noqa: E402


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _contains_terms(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def _first_relevant_rank(
    document_texts: list[str],
    terms: list[str],
) -> int | None:
    for index, text in enumerate(document_texts, start=1):
        if _contains_terms(text, terms):
            return index
    return None


def _citation_valid(route: str, citations: list[Any]) -> bool:
    if route == "refuse":
        return len(citations) == 0
    if not citations:
        return False
    return all(
        bool(citation.title)
        and bool(citation.url or citation.document_id)
        for citation in citations
    )


async def run(dataset_path: Path, limit: int | None) -> dict[str, Any]:
    cases = load_dataset(dataset_path)
    if limit:
        cases = cases[:limit]

    graph = await create_graph()
    results: list[dict[str, Any]] = []

    for case in cases:
        start = time.perf_counter()
        try:
            state = await graph.ainvoke(
                {"messages": [HumanMessage(content=case["query"])]}
            )
            latency_ms = (time.perf_counter() - start) * 1000
            answer = state["messages"][-1].content
            route = state.get("route", "")
            citations = state.get("citations", [])
            documents = state.get("documents", [])
            web_documents = state.get("web_documents", [])
            error = None
        except Exception as exc:  # pragma: no cover - external service failure
            latency_ms = (time.perf_counter() - start) * 1000
            answer = ""
            route = "error"
            citations = []
            documents = []
            web_documents = []
            error = str(exc)

        document_texts = [document.content for document in documents]
        document_texts.extend(
            document.text or document.snippet for document in web_documents
        )
        retrieval_text = "\n".join(document_texts)
        expected_terms = case.get("expected_terms", [])
        rank = _first_relevant_rank(document_texts, expected_terms)
        answer_lower = answer.lower()
        answer_coverage = (
            sum(term.lower() in answer_lower for term in expected_terms)
            / len(expected_terms)
            if expected_terms
            else 1.0
        )

        results.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expected_route": case.get("expected_route"),
                "route": route,
                "route_correct": route == case.get("expected_route"),
                "latency_ms": round(latency_ms, 1),
                "answer": answer,
                "error": error,
                "citations": [asdict(citation) for citation in citations],
                "citation_count": len(citations),
                "citation_valid": _citation_valid(route, citations),
                "refusal_correct": (
                    route == "refuse"
                    if case.get("expected_route") == "refuse"
                    else route != "refuse"
                ),
                "expected_terms": expected_terms,
                "retrieval_text_present": bool(retrieval_text.strip()),
                "retrieval_hit": bool(expected_terms)
                and _contains_terms(retrieval_text, expected_terms),
                "retrieval_rank": rank,
                "reciprocal_rank": (1.0 / rank) if rank else 0.0,
                "answer_term_coverage": round(answer_coverage, 4),
                "documents": [
                    {
                        "id": document.id,
                        "title": document.title,
                        "score": document.score,
                    }
                    for document in documents
                ],
                "web_documents": [
                    {"title": document.title, "url": document.url}
                    for document in web_documents
                ],
            }
        )

    total = len(results)
    latencies = [result["latency_ms"] for result in results]
    summary = {
        "cases": total,
        "routing_accuracy": round(
            sum(result["route_correct"] for result in results) / total, 4
        ),
        "citation_validity": round(
            sum(result["citation_valid"] for result in results) / total, 4
        ),
        "refusal_correctness": round(
            sum(result["refusal_correct"] for result in results) / total, 4
        ),
        "retrieval_recall_at_5": round(
            sum(result["retrieval_hit"] for result in results) / total, 4
        ),
        "mrr": round(
            sum(result["reciprocal_rank"] for result in results) / total, 4
        ),
        "answer_term_coverage": round(
            sum(result["answer_term_coverage"] for result in results) / total, 4
        ),
        "p50_latency_ms": round(statistics.median(latencies), 1),
        "p95_latency_ms": round(
            sorted(latencies)[max(0, int(total * 0.95) - 1)], 1
        ),
        "errors": sum(1 for result in results if result["error"]),
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_path),
        "summary": summary,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local RAG evaluation harness.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evals" / "dataset.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evals" / "reports" / "local-latest.json",
    )
    args = parser.parse_args()

    report = asyncio.run(run(args.dataset, args.limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(json.dumps(summary, indent=2))
    print(f"\nReport: {args.output}")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
