import pytest
from src.inverted_index import InvertedIndex
from src.bm25_scorer import BM25Scorer


@pytest.fixture
def populated_index():
    idx = InvertedIndex()
    # doc1: Short and very concentrated on python
    idx.add_document(
        doc_id="short_doc",
        title="Python Fast",
        content="Python Python code."
    )
    # doc2: Very long document with lots of other filler words
    idx.add_document(
        doc_id="long_doc",
        title="Software Engineering Overview",
        content="This document covers architecture databases caching deployment python and infrastructure."
    )
    # doc3: Unrelated
    idx.add_document(
        doc_id="other_doc",
        title="Linux Kernel",
        content="Operating systems manage hardware memory drivers and process scheduling."
    )
    return idx


def test_average_doc_length_calculation(populated_index):
    scorer = BM25Scorer(index=populated_index)
    avg_len = scorer.compute_average_doc_length()
    assert avg_len > 0.0


def test_bm25_scoring_penalizes_verbose_documents(populated_index):
    scorer = BM25Scorer(index=populated_index)
    scores = scorer.score_query(["python"])

    assert "short_doc" in scores
    assert "long_doc" in scores
    # Concentrated document must rank higher than verbose document under BM25
    assert scores["short_doc"] > scores["long_doc"]
    assert "other_doc" not in scores


def test_bm25_empty_query(populated_index):
    scorer = BM25Scorer(index=populated_index)
    scores = scorer.score_query([])
    assert scores == {}