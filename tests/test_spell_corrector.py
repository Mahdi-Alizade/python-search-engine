import pytest
from src.inverted_index import InvertedIndex
from src.spell_corrector import SpellCorrector


@pytest.fixture
def corpus_index():
    idx = InvertedIndex()
    idx.add_document(
        doc_id="d1",
        title="Python Frameworks",
        content="Python developers build modern backend architectures using FastAPI."
    )
    idx.add_document(
        doc_id="d2",
        title="Database Systems",
        content="PostgreSQL relational database handles transactions and replication."
    )
    return idx


def test_levenshtein_distance_calculation(corpus_index):
    corrector = SpellCorrector(index=corpus_index)
    dist1 = corrector.calculate_levenshtein_distance("kitten", "sitting")
    assert dist1 == 3

    dist2 = corrector.calculate_levenshtein_distance("python", "pythn")
    assert dist2 == 1

    dist3 = corrector.calculate_levenshtein_distance("fast", "fast")
    assert dist3 == 0


def test_suggest_term_single_typo(corpus_index):
    corrector = SpellCorrector(index=corpus_index)
    # 'pythn' should suggest 'python'
    suggestion = corrector.suggest_term("pythn")
    assert suggestion == "python"


def test_suggest_term_already_correct(corpus_index):
    corrector = SpellCorrector(index=corpus_index)
    suggestion = corrector.suggest_term("database")
    # 'database' stems to 'databas' or exists in vocab
    assert suggestion is not None


def test_suggest_full_query(corpus_index):
    corrector = SpellCorrector(index=corpus_index)
    # Query with misspelled word
    suggestion = corrector.suggest_query("fastapi pythn")
    assert suggestion is not None
    assert "python" in suggestion


def test_suggest_query_no_typos(corpus_index):
    corrector = SpellCorrector(index=corpus_index)
    suggestion = corrector.suggest_query("python database")
    # No corrections required
    assert suggestion is None