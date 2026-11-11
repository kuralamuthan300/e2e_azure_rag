from __future__ import annotations

from typing import Protocol

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import (
    APIConnectionError,
    APITimeoutError,
    AzureOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings

_RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be generated."""


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FoundryEmbeddings:
    """Generate embeddings with an Azure OpenAI / Foundry embedding deployment."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: AzureOpenAI | None = None,
    ) -> None:
        self._settings = settings
        self.dimensions = settings.embedding_dimensions
        self._client = client or self._build_client(settings)

    @staticmethod
    def _build_client(settings: Settings) -> AzureOpenAI:
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureOpenAI(
            azure_endpoint=settings.openai_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=settings.openai_api_version,
        )

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._settings.embedding_deployment,
            input=texts,
            dimensions=self._settings.embedding_dimensions,
        )
        return [item.embedding for item in response.data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                batch_vectors = self._embed_batch(batch)
            except Exception as exc:  # pragma: no cover - provider-specific failures
                raise EmbeddingError(
                    f"Embedding request failed for batch starting at {start}"
                ) from exc

            for vector in batch_vectors:
                if len(vector) != self.dimensions:
                    raise EmbeddingError(
                        "Embedding dimension mismatch: "
                        f"expected {self.dimensions}, got {len(vector)}. "
                        "Check AZURE_EMBEDDING_DIMENSIONS."
                    )
            vectors.extend(batch_vectors)

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Embedding count mismatch: requested {len(texts)}, got {len(vectors)}"
            )
        return vectors
