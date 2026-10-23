import pytest
from src.models import DocumentRecord, TermPosting


def test_document_record_serialization_roundtrip():
    doc = DocumentRecord(
        doc_id="doc-101",
        title="Microservice Architecture",
        content="Designing resilient backend services.",
        url="https://example.com/docs/101",
        total_tokens=42
    )

    data = doc.to_dict()
    assert data["doc_id"] == "doc-101"
    assert data["total_tokens"] == 42

    restored = DocumentRecord.from_dict(data)
    assert restored.doc_id == doc.doc_id
    assert restored.title == doc.title
    assert restored.url == doc.url
    assert restored.total_tokens == doc.total_tokens


def test_term_posting_add_occurrence():
    posting = TermPosting(doc_id="doc-202")
    assert posting.term_frequency == 0
    assert len(posting.positions) == 0

    posting.add_occurrence(5)
    posting.add_occurrence(18)

    assert posting.term_frequency == 2
    assert posting.positions == [5, 18]


def test_term_posting_serialization_roundtrip():
    posting = TermPosting(
        doc_id="doc-303",
        term_frequency=3,
        positions=[1, 10, 25]
    )

    data = posting.to_dict()
    assert data["term_frequency"] == 3

    restored = TermPosting.from_dict(data)
    assert restored.doc_id == "doc-303"
    assert restored.term_frequency == 3
    assert restored.positions == [1, 10, 25]