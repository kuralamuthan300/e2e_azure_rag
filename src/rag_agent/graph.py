from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from grader import ContextGrader
from llm import FoundryLLM
from models import Citation, ContextGrade, RetrievedDocument, WebDocument
from toolbox import ToolboxClient, ToolboxError

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 4000
MIN_GRADE_CONFIDENCE = 0.5
_REFUSAL_MARKERS = (
    "could not find enough evidence",
    "cannot find enough evidence",
    "can't find enough evidence",
    "does not contain",
    "do not contain",
    "does not provide",
    "do not provide",
    "does not say",
    "do not say",
    "not provided in the retrieved",
    "cannot determine",
    "can't determine",
    "cannot provide",
    "can't provide",
    "cannot disregard",
    "can't disregard",
    "unable to answer",
    "no reliable evidence",
    "i don't know",
    "i do not know",
)


def _is_refusal(answer: str) -> bool:
    # Only inspect the opening of the answer. Explanations of the refusal path
    # can legitimately mention phrases like "no reliable evidence" later on.
    prefix = answer.strip().lower()[:300]
    return any(marker in prefix for marker in _REFUSAL_MARKERS)


class RagState(MessagesState):
    """LangGraph state compatible with the Foundry Responses host."""

    question: str
    documents: list[RetrievedDocument]
    web_documents: list[WebDocument]
    grade: ContextGrade | None
    route: Literal["rag", "web", "hybrid", "refuse"]
    citations: list[Citation]
    answer: str
    errors: list[str]


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content) if content is not None else ""


def _document_context(documents: list[RetrievedDocument]) -> str:
    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        content = document.content[:MAX_CONTEXT_CHARS]
        blocks.append(f"[{index}] {document.title}\n{content}")
    return "\n\n".join(blocks)


def _web_context(documents: list[WebDocument]) -> str:
    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        content = (document.text or document.snippet)[:MAX_CONTEXT_CHARS]
        blocks.append(f"[{index}] {document.title}\nURL: {document.url}\n{content}")
    return "\n\n".join(blocks)


def extract_question(state: RagState) -> dict[str, object]:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return {
                "question": _content_to_text(message.content).strip(),
                "documents": [],
                "web_documents": [],
                "grade": None,
                "route": "refuse",
                "citations": [],
                "answer": "",
                "errors": [],
            }
    return {
        "question": "",
        "documents": [],
        "web_documents": [],
        "grade": None,
        "route": "refuse",
        "citations": [],
        "answer": "",
        "errors": ["No user message found"],
    }


async def retrieve_docs(state: RagState, toolbox: ToolboxClient) -> dict[str, object]:
    question = state.get("question", "")
    if not question:
        return {"documents": [], "errors": ["No question to retrieve"]}

    try:
        documents = await toolbox.search(question, top_k=5)
    except ToolboxError as exc:
        logger.warning("Document retrieval failed: %s", exc)
        return {"documents": [], "errors": [str(exc)]}

    logger.info("Retrieved %d document(s) for %r", len(documents), question)
    return {"documents": documents}


async def grade_context(
    state: RagState,
    grader: ContextGrader,
) -> dict[str, object]:
    grade = await grader.grade(
        state.get("question", ""),
        state.get("documents", []),
    )
    return {"grade": grade}


def route_after_grade(state: RagState) -> str:
    grade = state.get("grade")
    if (
        grade
        and grade.sufficient
        and grade.coverage == "full"
        and grade.confidence >= MIN_GRADE_CONFIDENCE
    ):
        return "answer"
    return "web"


async def web_search(state: RagState, toolbox: ToolboxClient) -> dict[str, object]:
    question = state.get("question", "")
    try:
        web_documents = await toolbox.web_search(question, max_results=3)
    except ToolboxError as exc:
        logger.warning("Web search failed: %s", exc)
        return {
            "web_documents": [],
            "errors": [*state.get("errors", []), str(exc)],
        }

    grade = state.get("grade")
    use_hybrid = bool(
        state.get("documents")
        and grade
        and grade.coverage == "partial"
    )
    route: Literal["web", "hybrid"] = "hybrid" if use_hybrid else "web"
    logger.info(
        "Retrieved %d web result(s) for %r (route=%s)",
        len(web_documents),
        question,
        route,
    )
    return {"web_documents": web_documents, "route": route}


def route_after_web(state: RagState) -> str:
    if not state.get("web_documents"):
        return "refuse"
    if state.get("route") == "hybrid" and state.get("documents"):
        return "hybrid"
    return "web"


async def _generate_answer(
    llm: FoundryLLM,
    *,
    question: str,
    context: str,
    source_kind: str,
) -> str:
    prompt = (
        "You are a grounded RAG assistant. If the supplied context does not "
        "contain the answer, say so explicitly.\n\n"
        f"Answer the question using only the {source_kind} below. Cite claims "
        "with bracketed numbers such as [1]. Do not use outside knowledge. Do not "
        "add a Sources section; the application appends it.\n\n"
        f"Question:\n{question}\n\n"
        f"{source_kind.capitalize()}:\n{context}"
    )
    return await llm.complete(prompt)


async def answer_from_docs(
    state: RagState,
    llm: FoundryLLM,
) -> dict[str, object]:
    documents = state.get("documents", [])
    citations = [
        Citation.from_document(index, document)
        for index, document in enumerate(documents, start=1)
    ]
    answer = await _generate_answer(
        llm,
        question=state.get("question", ""),
        context=_document_context(documents),
        source_kind="retrieved documents",
    )
    if _is_refusal(answer):
        return {"answer": answer, "citations": [], "route": "refuse"}
    return {"answer": answer, "citations": citations, "route": "rag"}


async def answer_from_web(
    state: RagState,
    llm: FoundryLLM,
) -> dict[str, object]:
    documents = state.get("web_documents", [])
    citations = [
        Citation.from_web_document(index, document)
        for index, document in enumerate(documents, start=1)
    ]
    answer = await _generate_answer(
        llm,
        question=state.get("question", ""),
        context=_web_context(documents),
        source_kind="web search results",
    )
    if _is_refusal(answer):
        return {"answer": answer, "citations": [], "route": "refuse"}
    return {"answer": answer, "citations": citations, "route": "web"}


async def answer_hybrid(
    state: RagState,
    llm: FoundryLLM,
) -> dict[str, object]:
    documents = state.get("documents", [])
    web_documents = state.get("web_documents", [])
    citations: list[Citation] = []
    blocks: list[str] = []

    for index, document in enumerate(documents, start=1):
        citations.append(Citation.from_document(index, document))
        blocks.append(
            f"[{index}] Document: {document.title}\n"
            f"{document.content[:MAX_CONTEXT_CHARS]}"
        )
    offset = len(documents)
    for index, web_document in enumerate(web_documents, start=offset + 1):
        citations.append(Citation.from_web_document(index, web_document))
        content = (web_document.text or web_document.snippet)[:MAX_CONTEXT_CHARS]
        blocks.append(
            f"[{index}] Web: {web_document.title}\n"
            f"URL: {web_document.url}\n{content}"
        )

    answer = await _generate_answer(
        llm,
        question=state.get("question", ""),
        context="\n\n".join(blocks),
        source_kind="retrieved documents and web search results",
    )
    if _is_refusal(answer):
        return {"answer": answer, "citations": [], "route": "refuse"}
    return {"answer": answer, "citations": citations, "route": "hybrid"}


def refuse(state: RagState) -> dict[str, object]:
    grade = state.get("grade")
    if grade and grade.reason:
        reason = grade.reason
    elif state.get("errors"):
        reason = "; ".join(state["errors"])
    else:
        reason = "No reliable evidence was found."
    return {
        "answer": (
            "I could not find enough evidence in the indexed documents or web "
            f"results to answer that question. {reason}"
        ),
        "citations": [],
        "route": "refuse",
    }


def finalize(state: RagState) -> dict[str, object]:
    answer = state.get("answer", "").strip()
    citations = state.get("citations", [])
    if citations:
        sources = "\n".join(citation.render() for citation in citations)
        answer = f"{answer}\n\nSources:\n{sources}"
    if not answer:
        answer = "I could not produce an answer from the retrieved evidence."
    return {"messages": [AIMessage(content=answer)]}


def build_graph(
    llm: FoundryLLM,
    toolbox: ToolboxClient,
    grader: ContextGrader,
) -> object:
    """Build the deterministic RAG graph consumed by the Foundry host."""

    graph = StateGraph(RagState)

    async def retrieve_node(state: RagState) -> dict[str, object]:
        return await retrieve_docs(state, toolbox)

    async def grade_node(state: RagState) -> dict[str, object]:
        return await grade_context(state, grader)

    async def web_node(state: RagState) -> dict[str, object]:
        return await web_search(state, toolbox)

    async def answer_docs_node(state: RagState) -> dict[str, object]:
        return await answer_from_docs(state, llm)

    async def answer_web_node(state: RagState) -> dict[str, object]:
        return await answer_from_web(state, llm)

    async def answer_hybrid_node(state: RagState) -> dict[str, object]:
        return await answer_hybrid(state, llm)

    graph.add_node("extract_question", extract_question)
    graph.add_node("retrieve_docs", retrieve_node)
    graph.add_node("grade_context", grade_node)
    graph.add_node("web_search", web_node)
    graph.add_node("answer_from_docs", answer_docs_node)
    graph.add_node("answer_from_web", answer_web_node)
    graph.add_node("answer_hybrid", answer_hybrid_node)
    graph.add_node("refuse", refuse)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "extract_question")
    graph.add_edge("extract_question", "retrieve_docs")
    graph.add_edge("retrieve_docs", "grade_context")
    graph.add_conditional_edges(
        "grade_context",
        route_after_grade,
        {
            "answer": "answer_from_docs",
            "web": "web_search",
        },
    )
    graph.add_conditional_edges(
        "web_search",
        route_after_web,
        {
            "web": "answer_from_web",
            "hybrid": "answer_hybrid",
            "refuse": "refuse",
        },
    )
    graph.add_edge("answer_from_docs", "finalize")
    graph.add_edge("answer_from_web", "finalize")
    graph.add_edge("answer_hybrid", "finalize")
    graph.add_edge("refuse", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
