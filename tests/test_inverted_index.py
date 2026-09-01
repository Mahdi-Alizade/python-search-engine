import os
import pytest
from src.inverted_index import InvertedIndex


@pytest.fixture
def empty_index():
    return InvertedIndex()


@pytest.fixture
def populated_index():
    idx = InvertedIndex()
    idx.add_document(
        doc_id="doc1",
        title="Python Backend Guide",
        content="Python is great for backend APIs and distributed systems.",
        url="https://example.com/python-backend"
    )
    idx.add_document(
        doc_id="doc2",
        title="Distributed Databases",
        content="Scalable databases support high throughput and fault tolerance.",
        url="https://example.com/databases"
    )
    idx.add_document(
        doc_id="doc3",
        title="Python Performance",
        content="Optimizing Python code with asynchronous tasks and caching.",
        url="https://example.com/python-perf"
    )
    return idx


def test_add_document_and_total_count(empty_index):
    assert empty_index.total_documents == 0
    empty_index.add_document("doc1", "Fast Search", "Building a custom search engine.")
    assert empty_index.total_documents == 1
    doc = empty_index.get_document("doc1")
    assert doc is not None
    assert doc.title == "Fast Search"
    assert doc.token_count > 0


def test_get_postings_term_presence(populated_index):
    python_postings = populated_index.get_postings("python")
    assert "doc1" in python_postings
    assert "doc3" in python_postings
    assert "doc2" not in python_postings
    assert python_postings["doc1"].term_frequency >= 1


def test_get_postings_nonexistent_term(populated_index):
    postings = populated_index.get_postings("nonexistentword123")
    assert postings == {}


def test_save_and_load_persistence(populated_index, tmp_path):
    storage_file = os.path.join(tmp_path, "storage", "test_index.json")

    # Save to disk
    populated_index.save_to_file(storage_file)
    assert os.path.exists(storage_file)

    # Load into fresh instance
    new_index = InvertedIndex()
    new_index.load_from_file(storage_file)

    assert new_index.total_documents == populated_index.total_documents
    assert new_index.get_document("doc1").title == "Python Backend Guide"

    postings = new_index.get_postings("databases")
    assert "doc2" in postings
    assert postings["doc2"].term_frequency >= 1