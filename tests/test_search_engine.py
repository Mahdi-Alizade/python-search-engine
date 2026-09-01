import pytest
from src.inverted_index import InvertedIndex
from src.search_engine import SearchEngine


@pytest.fixture
def engine():
    idx = InvertedIndex()
    idx.add_document(
        doc_id="doc1",
        title="Python Asynchronous Programming",
        content="Asynchronous programming in Python uses asyncio and event loops for high performance.",
        url="https://docs.python.org/asyncio"
    )
    idx.add_document(
        doc_id="doc2",
        title="Database Indexing Basics",
        content="Indexes improve query speed in relational databases like PostgreSQL and MySQL.",
        url="https://example.com/db-indexing"
    )
    idx.add_document(
        doc_id="doc3",
        title="Building Search Engines with Python",
        content="Learn how to build full text search engines in Python using inverted indexes and ranking algorithms.",
        url="https://example.com/python-search"
    )
    return SearchEngine(index=idx)


def test_search_empty_query(engine):
    results = engine.search("")
    assert results == []


def test_search_single_keyword(engine):
    results = engine.search("PostgreSQL")
    assert len(results) == 1
    assert results[0].doc_id == "doc2"
    assert results[0].score > 0
    assert "PostgreSQL" in results[0].snippet


def test_search_ranking_relevance(engine):
    results = engine.search("Python search engines")
    assert len(results) >= 2
    # doc3 has both 'python', 'search', 'engine' -> should rank first
    assert results[0].doc_id == "doc3"
    assert results[0].score > results[1].score


def test_search_non_matching_query(engine):
    results = engine.search("quantum mechanics physics")
    assert results == []


def test_search_top_k_limit(engine):
    results = engine.search("Python", top_k=1)
    assert len(results) == 1