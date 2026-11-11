from __future__ import annotations

from rag_ingest.index import VECTOR_PROFILE_NAME, build_index


def test_index_has_required_fields() -> None:
    index = build_index("rag-docs", dimensions=1536)
    fields = {field.name: field for field in index.fields}

    assert fields["id"].key is True
    assert fields["content"].searchable is True
    assert fields["content_vector"].vector_search_dimensions == 1536
    assert fields["content_vector"].vector_search_profile_name == VECTOR_PROFILE_NAME
    assert fields["content_vector"].retrievable is False
    assert fields["source"].filterable is True


def test_index_has_vector_profile() -> None:
    index = build_index("rag-docs", dimensions=768)

    assert index.vector_search is not None
    assert index.vector_search.profiles[0].name == VECTOR_PROFILE_NAME
    assert index.vector_search.profiles[0].algorithm_configuration_name == "hnsw"
