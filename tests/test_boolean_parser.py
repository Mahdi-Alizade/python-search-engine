import pytest
from src.inverted_index import InvertedIndex
from src.boolean_parser import BooleanQueryEngine


@pytest.fixture
def boolean_index():
    idx = InvertedIndex()
    idx.add_document(
        doc_id="doc1",
        title="Python Basics",
        content="Python is an interpreted programming language."
    )
    idx.add_document(
        doc_id="doc2",
        title="Python and Backend",
        content="Python is widely used in backend API development with FastAPI."
    )
    idx.add_document(
        doc_id="doc3",
        title="Frontend Architecture",
        content="React and TypeScript for web applications."
    )
    return idx


def test_boolean_and_operator(boolean_index):
    engine = BooleanQueryEngine(index=boolean_index)
    results = engine.evaluate("python AND backend")

    # Only doc2 contains both 'python' and 'backend'
    assert results == {"doc2"}


def test_boolean_or_operator(boolean_index):
    engine = BooleanQueryEngine(index=boolean_index)
    results = engine.evaluate("backend OR frontend")

    # doc2 has backend, doc3 has frontend
    assert "doc2" in results
    assert "doc3" in results
    assert "doc1" not in results


def test_boolean_not_operator(boolean_index):
    engine = BooleanQueryEngine(index=boolean_index)
    # Documents having python but excluding backend
    results = engine.evaluate("python NOT backend")

    assert "doc1" in results
    assert "doc2" not in results


def test_boolean_empty_query(boolean_index):
    engine = BooleanQueryEngine(index=boolean_index)
    results = engine.evaluate("")
    assert results == set()