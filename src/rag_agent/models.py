from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    """A document returned by the Azure AI Search toolbox tool."""

    id: str
    title: str
    content: str
    url: str | None = None
    source: str | None = None
    score: float | None = None

    @classmethod
    def from_tool_document(cls, document: dict[str, Any]) -> RetrievedDocument:
        score = document.get("score")
        return cls(
            id=str(document.get("id") or ""),
            title=str(document.get("title") or "Untitled document"),
            content=str(document.get("content") or ""),
            url=str(document["url"]) if document.get("url") else None,
            source=str(document["filepath"]) if document.get("filepath") else None,
            score=float(score) if score is not None else None,
        )


@dataclass(frozen=True, slots=True)
class WebDocument:
    """A result returned by the web-search MCP tool."""

    title: str
    url: str
    snippet: str = ""
    text: str = ""

    @classmethod
    def from_tool_document(cls, document: dict[str, Any]) -> WebDocument:
        return cls(
            title=str(document.get("title") or document.get("url") or "Web result"),
            url=str(document.get("url") or ""),
            snippet=str(document.get("snippet") or ""),
            text=str(document.get("text") or ""),
        )


@dataclass(frozen=True, slots=True)
class Citation:
    """A source citation derived from a retrieved document."""

    number: int
    title: str
    document_id: str
    url: str | None = None
    source: str | None = None
    score: float | None = None

    @classmethod
    def from_document(
        cls,
        number: int,
        document: RetrievedDocument,
    ) -> Citation:
        return cls(
            number=number,
            title=document.title,
            document_id=document.id,
            url=document.url,
            source=document.source,
            score=document.score,
        )

    @classmethod
    def from_web_document(cls, number: int, document: WebDocument) -> Citation:
        return cls(
            number=number,
            title=document.title,
            document_id=document.url,
            url=document.url,
            source=document.url,
        )

    def render(self) -> str:
        details = [f"[{self.number}] {self.title}"]
        if self.url:
            details.append(f"URL: {self.url}")
        if self.source and self.source != self.url:
            details.append(f"Source: {self.source}")
        if self.document_id and self.document_id != self.url:
            details.append(f"ID: {self.document_id}")
        return " | ".join(details)


class ContextGrade(BaseModel):
    """Structured result of the context sufficiency check."""

    sufficient: bool = Field(
        description="True when the retrieved documents contain enough evidence to answer."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that the retrieved context is sufficient.",
    )
    coverage: Literal["none", "partial", "full"] = Field(
        description=(
            "How much of the question the retrieved documents cover: none, "
            "partial, or full."
        )
    )
    reason: str = Field(description="Short explanation for the decision.")
