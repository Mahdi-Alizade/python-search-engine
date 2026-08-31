import pytest
from src.text_processor import TextProcessor


@pytest.fixture
def processor():
    return TextProcessor()


def test_tokenize_simple(processor):
    text = "Hello World! Fast and scalable search engine."
    tokens = processor.tokenize(text)
    assert tokens == ["hello", "world", "fast", "and", "scalable", "search", "engine"]


def test_remove_stopwords(processor):
    tokens = ["this", "is", "a", "scalable", "distributed", "system"]
    filtered = processor.remove_stopwords(tokens)
    assert filtered == ["scalable", "distributed", "system"]


def test_stemming(processor):
    assert processor.stem("running") == "runn"
    assert processor.stem("connections") == "connection"
    assert processor.stem("developer") == "developer"
    assert processor.stem("searchable") == "search"


def test_full_pipeline_process(processor):
    text = "The quick brown foxes were jumping over lazy dogs!"
    result = processor.process(text)
    assert "quick" in result
    assert "brown" in result
    assert "fox" in result
    assert "lazy" in result
    assert "the" not in result
    assert "were" not in result