from __future__ import annotations

from models import Citation, ContextGrade, RetrievedDocument, WebDocument


def test_document_citation_renders_url_and_source() -> None:
    document = RetrievedDocument(
        id="chunk-1",
        title="Refund policy",
        content="Refunds take 30 days.",
        url="https://example.com/refunds",
        source="refunds.md",
        score=0.9,
    )
    citation = Citation.from_document(1, document)

    rendered = citation.render()

    assert rendered.startswith("[1] Refund policy")
    assert "URL: https://example.com/refunds" in rendered
    assert "Source: refunds.md" in rendered
    assert "ID: chunk-1" in rendered


def test_web_citation_uses_url_as_identifier() -> None:
    citation = Citation.from_web_document(
        2,
        WebDocument(
            title="Refund guidance",
            url="https://contoso.example/refunds",
            snippet="Refunds take 30 days.",
        ),
    )

    assert citation.document_id == "https://contoso.example/refunds"
    assert citation.url == "https://contoso.example/refunds"
    assert "ID:" not in citation.render()


def test_context_grade_rejects_out_of_range_confidence() -> None:
    try:
        ContextGrade(
            sufficient=True,
            confidence=1.5,
            coverage="full",
            reason="invalid",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("ContextGrade accepted confidence above 1.0")
