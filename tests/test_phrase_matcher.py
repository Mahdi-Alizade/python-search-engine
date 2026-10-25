import pytest
from src.inverted_index import InvertedIndex
from src.phrase_matcher import PhraseMatcher


@pytest.fixture
def index_with_positions():
    idx = InvertedIndex()
    # doc1: Exact phrase 'distributed systems' exists consecutively
    idx.add_document(
        doc_id="doc1",
        title="Distributed Systems Guide",
        content="We build distributed systems using microservices."
    )
    # doc2: Contains both words but separated / out of order
    idx.add_document(
        doc_id="doc2",
        title="Systems Engineering",
        content="Systems that are highly reliable and distributed worldwide."
    )
    # doc3: Unrelated
    idx.add_document(
        doc_id="doc3",
        title="Web Design",
        content="Frontend interface and modern responsive CSS layout."
    )
    return idx


def test_phrase_exact_sequence_found(index_with_positions):
    matcher = PhraseMatcher(index=index_with_positions)
    matches = matcher.find_phrase_matches("distributed systems")

    # doc1 must match because 'distributed systems' is consecutive
    assert "doc1" in matches
    # doc2 contains words separated by other tokens, so it must not match
    assert "doc2" not in matches
    assert "doc3" not in matches


def test_phrase_single_word(index_with_positions):
    matcher = PhraseMatcher(index=index_with_positions)
    matches = matcher.find_phrase_matches("microservices")
    assert matches == ["doc1"]


def test_phrase_non_existent(index_with_positions):
    matcher = PhraseMatcher(index=index_with_positions)
    matches = matcher.find_phrase_matches("quantum computing algorithms")
    assert matches == []