from __future__ import annotations

from dataclasses import dataclass, field

from models import ContextGrade, RetrievedDocument, WebDocument
from toolbox import ToolboxError


@dataclass
class FakeLLM:
    answer: str = "Grounded answer [1]."
    prompts: list[str] = field(default_factory=list)

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


@dataclass
class FakeToolbox:
    documents: list[RetrievedDocument] = field(default_factory=list)
    web_documents: list[WebDocument] = field(default_factory=list)
    search_error: Exception | None = None
    web_error: Exception | None = None

    async def search(self, query: str, *, top_k: int = 5) -> list[RetrievedDocument]:
        if self.search_error is not None:
            raise self.search_error
        return self.documents[:top_k]

    async def web_search(self, query: str, *, max_results: int = 3) -> list[WebDocument]:
        if self.web_error is not None:
            raise self.web_error
        return self.web_documents[:max_results]


@dataclass
class FakeGrader:
    result: ContextGrade = field(
        default_factory=lambda: ContextGrade(
            sufficient=True,
            confidence=0.9,
            coverage="full",
            reason="The documents answer the question.",
        )
    )
    error: Exception | None = None

    async def grade(
        self,
        question: str,
        documents: list[RetrievedDocument],
    ) -> ContextGrade:
        if self.error is not None:
            raise self.error
        return self.result


def document(
    document_id: str = "chunk-1",
    *,
    title: str = "Refund policy",
    content: str = "Refunds are available within 30 days.",
) -> RetrievedDocument:
    return RetrievedDocument(
        id=document_id,
        title=title,
        content=content,
        url="https://example.com/refunds",
        source="refunds.md",
        score=0.91,
    )


def web_document(
    *,
    title: str = "Refund guidance",
    url: str = "https://contoso.example/refunds",
    text: str = "Refunds are available within 30 days.",
) -> WebDocument:
    return WebDocument(title=title, url=url, snippet=text[:80], text=text)


def toolbox_error(message: str = "toolbox unavailable") -> ToolboxError:
    return ToolboxError(message)
