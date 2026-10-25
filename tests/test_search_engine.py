import pytest
from src.inverted_index import InvertedIndex
from src.search_engine import SearchEngine


@pytest.fixture
def test_search_engine():
    idx = InvertedIndex()
    idx.add_document(
        doc_id="doc_py_async",
        title="Python Asynchronous Programming",
        content="Asynchronous programming in Python uses asyncio and event loops for high performance.",
        url="https://docs.python.org/asyncio"
    )
    idx.add_document(
        doc_id="doc_db_index",
        title="Database Indexing Basics",
        content="Indexes improve query speed in relational databases like PostgreSQL and MySQL.",
        url="https://example.com/db-indexing"
    )
    idx.add_document(
        doc_id="doc_search_core",
        title="Building Search Engines with Python",
        content="Learn how to build full text search engines in Python using inverted indexes and ranking algorithms.",
        url="https://example.com/python-search"
    )
    engine = SearchEngine(index=idx)
    return engine


def test_search_empty_query(test_search_engine):
    results = test_search_engine.search("")
    assert results == []


def test_search_single_keyword(test_search_engine):
    results = test_search_engine.search("PostgreSQL")
    assert len(results) == 1
    assert results[0].doc_id == "doc_db_index"
    assert results[0].score > 0
    assert "PostgreSQL" in results[0].snippet


def test_search_ranking_relevance(test_search_engine):
    results = test_search_engine.search("Python search engines")
    assert len(results) >= 2
    # Document doc_search_core contains all matching keywords and ranks on top
    assert results[0].doc_id == "doc_search_core"
    assert results[0].score > results[1].score


def test_search_exact_phrase_matching(test_search_engine):
    # Search for an exact phrase that only exists sequentially in doc_py_async
    results = test_search_engine.search('"asynchronous programming"')
    assert len(results) == 1
    assert results[0].doc_id == "doc_py_async"


def test_search_phrase_no_match(test_search_engine):
    # Terms exist in the corpus but not sequentially together
    results = test_search_engine.search('"programming relational"')
    assert results == []


def test_search_non_matching_query(test_search_engine):
    results = test_search_engine.search("quantum mechanics physics")
    assert results == []


def test_search_top_k_limit(test_search_engine):
    results = test_search_engine.search("Python", top_k=1)
    assert len(results) == 1