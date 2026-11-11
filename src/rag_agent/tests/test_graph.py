from __future__ import annotations

from fakes import (
    FakeGrader,
    FakeLLM,
    FakeToolbox,
    document,
    toolbox_error,
    web_document,
)
from langchain_core.messages import AIMessage, HumanMessage

from graph import build_graph
from models import ContextGrade


async def run_graph(
    *,
    question: str = "What is the refund policy?",
    documents: list | None = None,
    web_documents: list | None = None,
    grade: ContextGrade | None = None,
    answer: str = "Refunds are available within 30 days [1].",
    search_error: Exception | None = None,
    web_error: Exception | None = None,
) -> dict:
    llm = FakeLLM(answer=answer)
    toolbox = FakeToolbox(
        documents=documents or [],
        web_documents=web_documents or [],
        search_error=search_error,
        web_error=web_error,
    )
    grader = FakeGrader(
        result=grade
        or ContextGrade(
            sufficient=True,
            confidence=0.9,
            coverage="full",
            reason="The documents answer the question.",
        )
    )
    graph = build_graph(llm, toolbox, grader)
    return await graph.ainvoke({"messages": [HumanMessage(content=question)]})


async def test_rag_route_cites_documents() -> None:
    result = await run_graph(documents=[document()])

    assert result["route"] == "rag"
    assert len(result["citations"]) == 1
    assert result["citations"][0].document_id == "chunk-1"
    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert "Sources:" in final.content
    assert "https://example.com/refunds" in final.content


async def test_web_route_cites_urls() -> None:
    result = await run_graph(
        documents=[],
        web_documents=[web_document(url="https://example.org/page")],
        grade=ContextGrade(
            sufficient=False,
            confidence=0.2,
            coverage="none",
            reason="No document evidence.",
        ),
        answer="The web result says refunds take 30 days [1].",
    )

    assert result["route"] == "web"
    assert len(result["citations"]) == 1
    assert result["citations"][0].url == "https://example.org/page"
    assert "Sources:" in result["messages"][-1].content


async def test_hybrid_route_cites_documents_and_web() -> None:
    result = await run_graph(
        documents=[document()],
        web_documents=[web_document()],
        grade=ContextGrade(
            sufficient=False,
            confidence=0.7,
            coverage="partial",
            reason="Documents cover the policy but not the current processing time.",
        ),
        answer="Documents and a current web source both describe refunds [1][2].",
    )

    assert result["route"] == "hybrid"
    assert [citation.document_id for citation in result["citations"]] == [
        "chunk-1",
        "https://contoso.example/refunds",
    ]
    final = result["messages"][-1].content
    assert "[1]" in final
    assert "[2]" in final


async def test_refuses_when_no_evidence() -> None:
    result = await run_graph(
        documents=[],
        web_documents=[],
        grade=ContextGrade(
            sufficient=False,
            confidence=0.0,
            coverage="none",
            reason="No evidence was retrieved.",
        ),
    )

    assert result["route"] == "refuse"
    assert result["citations"] == []
    assert "could not find enough evidence" in result["messages"][-1].content


async def test_retrieval_error_falls_back_to_web() -> None:
    result = await run_graph(
        documents=[],
        web_documents=[web_document()],
        grade=ContextGrade(
            sufficient=False,
            confidence=0.0,
            coverage="none",
            reason="Retrieval failed.",
        ),
        search_error=toolbox_error(),
        answer="A web source describes refunds [1].",
    )

    assert result["route"] == "web"
    assert "toolbox unavailable" in " ".join(result["errors"])
    assert len(result["citations"]) == 1


async def test_web_error_refuses() -> None:
    result = await run_graph(
        documents=[],
        grade=ContextGrade(
            sufficient=False,
            confidence=0.0,
            coverage="none",
            reason="No document evidence.",
        ),
        web_error=toolbox_error("web search unavailable"),
    )

    assert result["route"] == "refuse"
    assert result["citations"] == []
    assert "web search unavailable" in " ".join(result["errors"])


async def test_refusal_marker_removes_citations() -> None:
    result = await run_graph(
        documents=[document()],
        answer="I could not find enough evidence in the retrieved documents to answer.",
    )

    assert result["route"] == "refuse"
    assert result["citations"] == []


async def test_no_human_message_refuses() -> None:
    llm = FakeLLM()
    grader = FakeGrader(
        result=ContextGrade(
            sufficient=False,
            confidence=0.0,
            coverage="none",
            reason="No question was provided.",
        )
    )
    graph = build_graph(llm, FakeToolbox(), grader)

    result = await graph.ainvoke({"messages": []})

    assert result["route"] == "refuse"
    assert "could not find enough evidence" in result["messages"][-1].content
