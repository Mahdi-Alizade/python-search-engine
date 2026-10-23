import os
import pytest
from src.inverted_index import InvertedIndex


@pytest.fixture
def empty_index():
    idx = InvertedIndex()
    return idx


@pytest.fixture
def sample_index():
    idx = InvertedIndex()
    idx.add_document(
        doc_id="doc_a",
        title="Fast Database Engines",
        content="Distributed databases store partitioned records across clusters.",
        url="https://example.com/db"
    )
    idx.add_document(
        doc_id="doc_b",
        title="Python Full Text Search",
        content="Building an inverted index in Python for fast search queries.",
        url="https://example.com/search"
    )
    return idx


def test_add_document_increases_count(empty_index):
    assert empty_index.total_documents == 0
    empty_index.add_document("doc_1", "Indexing Systems", "Inverted indexes allow instant lookup.")
    assert empty_index.total_documents == 1

    doc = empty_index.get_document("doc_1")
    assert doc is not None
    assert doc.title == "Indexing Systems"
    assert doc.total_tokens > 0


def test_get_postings_returns_valid_postings(sample_index):
    search_postings = sample_index.get_postings("search")
    assert "doc_b" in search_postings
    assert "doc_a" not in search_postings

    posting = search_postings["doc_b"]
    assert posting.term_frequency >= 1
    assert len(posting.positions) >= 1


def test_get_postings_missing_term(sample_index):
    postings = sample_index.get_postings("nonexistentterm999")
    assert postings == {}


def test_save_and_load_disk_persistence(sample_index, tmp_path):
    storage_path = os.path.join(tmp_path, "storage", "index_dump.json")

    # Persist current index
    sample_index.save_to_disk(storage_path)
    assert os.path.exists(storage_path)

    # Reconstruct inside a new index instance
    new_index = InvertedIndex()
    new_index.load_from_disk(storage_path)

    assert new_index.total_documents == 2
    assert new_index.get_document("doc_a").title == "Fast Database Engines"

    postings = new_index.get_postings("search")
    assert "doc_b" in postings
    assert postings["doc_b"].term_frequency == sample_index.get_postings("search")["doc_b"].term_frequency