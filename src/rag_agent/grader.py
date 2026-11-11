from __future__ import annotations

import asyncio
import logging
from typing import Any

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from models import ContextGrade, RetrievedDocument
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_AZURE_AI_SCOPE = "https://ai.azure.com/.default"
MAX_DOCUMENT_CHARS = 4000


class ContextGrader:
    """Grade retrieval sufficiency without going through LangChain.

    The Foundry hosting layer streams every LangChain model call in the graph.
    Using the OpenAI SDK directly keeps the grader's output out of the response
    stream while still using the same Foundry model deployment.
    """

    def __init__(
        self,
        project_endpoint: str,
        model_deployment: str,
        *,
        credential: TokenCredential | None = None,
    ) -> None:
        self._project_endpoint = project_endpoint.rstrip("/")
        self._model_deployment = model_deployment
        self._credential = credential or DefaultAzureCredential()

    async def grade(
        self,
        question: str,
        documents: list[RetrievedDocument],
    ) -> ContextGrade:
        if not documents:
            return ContextGrade(
                sufficient=False,
                confidence=0.0,
                coverage="none",
                reason="Azure AI Search returned no documents.",
            )

        context = "\n\n".join(
            f"[{index}] {document.title}\n{document.content[:MAX_DOCUMENT_CHARS]}"
            for index, document in enumerate(documents, start=1)
        )
        prompt = (
            "Return JSON only. Decide how much of the user's question the "
            "retrieved documents answer. Do not answer the question.\n\n"
            "Use coverage=full when the documents explicitly contain the answer "
            "to every part of the question. Use coverage=partial when some "
            "required fact is missing, such as an external URL, a current fact, "
            "or a second part of a multi-part question. Use coverage=none when "
            "the documents are unrelated. A question that mentions web search "
            "does not by itself make the coverage partial if the documents "
            "explain the requested behavior.\n\n"
            "Schema: {\"sufficient\": boolean, \"confidence\": number from 0 to 1, "
            "\"coverage\": \"none\" | \"partial\" | \"full\", \"reason\": string}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved documents:\n{context}"
        )

        try:
            token = await asyncio.to_thread(
                self._credential.get_token,
                _AZURE_AI_SCOPE,
            )
            client = AsyncOpenAI(
                base_url=f"{self._project_endpoint}/openai/v1",
                api_key=token.token,
            )
            response = await client.responses.create(
                model=self._model_deployment,
                input=prompt,
            )
            grade = ContextGrade.model_validate_json(_extract_json(response.output_text))
            return grade
        except Exception as exc:
            logger.warning("Direct context grading failed, using fallback: %s", exc)
            top_score = max(
                (
                    document.score
                    for document in documents
                    if document.score is not None
                ),
                default=0.0,
            )
            return ContextGrade(
                sufficient=top_score > 0,
                confidence=0.6 if top_score > 0 else 0.0,
                coverage="partial" if top_score > 0 else "none",
                reason=f"Grader unavailable; retrieval score={top_score:.3f}.",
            )


def _extract_json(text: str) -> str:
    """Extract a JSON object from a model response."""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Grader response did not contain JSON: {text[:200]}")
    return text[start : end + 1]


def parse_grade_payload(payload: dict[str, Any]) -> ContextGrade:
    """Parse a grade payload for tests."""

    return ContextGrade.model_validate(payload)
