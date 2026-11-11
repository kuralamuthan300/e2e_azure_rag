from __future__ import annotations

from contextlib import suppress
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

VECTOR_PROFILE_NAME = "default-profile"
HNSW_CONFIG_NAME = "hnsw"


class IndexingError(RuntimeError):
    """Raised when index creation or document upload fails."""


def build_index(name: str, dimensions: int) -> SearchIndex:
    """Build the Azure AI Search index definition used by the agent."""

    if dimensions <= 0:
        raise ValueError("dimensions must be greater than zero")

    return SearchIndex(
        name=name,
        fields=[
            SimpleField(name="id", type="Edm.String", key=True),
            SimpleField(name="doc_id", type="Edm.String", filterable=True),
            SimpleField(name="chunk_id", type="Edm.String", filterable=True),
            SearchableField(name="title", type="Edm.String"),
            SearchableField(name="content", type="Edm.String"),
            SearchField(
                name="content_vector",
                type="Collection(Edm.Single)",
                searchable=True,
                vector_search_dimensions=dimensions,
                vector_search_profile_name=VECTOR_PROFILE_NAME,
                retrievable=False,
            ),
            SimpleField(name="source", type="Edm.String", filterable=True),
            SimpleField(name="url", type="Edm.String"),
            SimpleField(
                name="chunk_index",
                type="Edm.Int32",
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="last_updated",
                type="Edm.DateTimeOffset",
                filterable=True,
                sortable=True,
            ),
            SimpleField(name="metadata", type="Edm.String"),
        ],
        vector_search=VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name=VECTOR_PROFILE_NAME,
                    algorithm_configuration_name=HNSW_CONFIG_NAME,
                )
            ],
            algorithms=[HnswAlgorithmConfiguration(name=HNSW_CONFIG_NAME)],
        ),
    )


class AzureSearchIndex:
    """Thin wrapper around the Azure AI Search data and index clients."""

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        *,
        credential: Any | None = None,
    ) -> None:
        credential = credential or DefaultAzureCredential()
        self.index_name = index_name
        self._index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
        self._search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential,
        )

    def ensure_index(self, dimensions: int, *, recreate: bool = False) -> None:
        """Create or update the index definition."""

        if recreate:
            with suppress(Exception):
                self._index_client.delete_index(self.index_name)

        index = build_index(self.index_name, dimensions)

        # Preserve vectorizer configuration added separately for hybrid search.
        # Re-ingestion must not silently downgrade the index to keyword-only.
        if not recreate:
            with suppress(Exception):
                existing = self._index_client.get_index(self.index_name)
                existing_vector_search = existing.vector_search
                new_vector_search = index.vector_search
                if existing_vector_search and new_vector_search:
                    new_vector_search.vectorizers = existing_vector_search.vectorizers
                    existing_profiles = {
                        profile.name: profile
                        for profile in (existing_vector_search.profiles or [])
                    }
                    for profile in new_vector_search.profiles or []:
                        old_profile = existing_profiles.get(profile.name)
                        if old_profile and old_profile.vectorizer_name:
                            profile.vectorizer_name = old_profile.vectorizer_name

        try:
            self._index_client.create_or_update_index(index)
        except Exception as exc:
            raise IndexingError(f"Could not create or update index {self.index_name!r}") from exc

    def upload(
        self,
        documents: list[dict[str, Any]],
        *,
        batch_size: int = 100,
    ) -> int:
        """Upload documents in batches and fail loudly on any rejected document."""

        if not documents:
            return 0
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        uploaded = 0
        failures: list[str] = []
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            try:
                results = self._search_client.upload_documents(documents=batch)
            except Exception as exc:
                raise IndexingError(
                    f"Azure AI Search upload failed for batch starting at {start}"
                ) from exc

            for result in results:
                if result.succeeded:
                    uploaded += 1
                else:
                    failures.append(
                        f"{result.key}: status={result.status_code} "
                        f"error={result.error_message}"
                    )

        if failures:
            preview = "\n".join(failures[:10])
            raise IndexingError(
                f"{len(failures)} document(s) were rejected by Azure AI Search:\n{preview}"
            )
        return uploaded

    def count(self) -> int:
        results = self._search_client.search(
            search_text="*",
            include_total_count=True,
            top=1,
        )
        return int(results.get_count() or 0)
