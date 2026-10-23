import pytest
from src.tokenizer import TextTokenizer


@pytest.fixture
def tokenizer():
    # Instantiate default tokenizer fixture for tests
    tok = TextTokenizer()
    return tok


def test_normalize_lowercase_and_strip(tokenizer):
    raw_text = "   HELLO Distributed WORLD!   "
    cleaned = tokenizer.normalize(raw_text)
    assert cleaned == "hello distributed world!"


def test_extract_tokens_splits_words_correctly(tokenizer):
    sentence = "Python-based search engine, version 1.0!"
    tokens = tokenizer.extract_tokens(sentence)
    # Checks that words are extracted cleanly without punctuation noise
    assert "python" in tokens
    assert "based" in tokens
    assert "search" in tokens
    assert "engine" in tokens


def test_filter_tokens_removes_stopwords_and_short_chars(tokenizer):
    raw_tokens = ["this", "is", "a", "fast", "and", "scalable", "database", "system"]
    filtered = tokenizer.filter_tokens(raw_tokens)
    # Stopwords like 'this', 'is', 'a', 'and' should be removed
    assert "scalable" in filtered
    assert "database" in filtered
    assert "system" in filtered
    assert "this" not in filtered
    assert "a" not in filtered


def test_stem_word_suffix_reduction(tokenizer):
    # Testing plural stripping
    assert tokenizer.stem_word("indexes") == "index"
    assert tokenizer.stem_word("servers") == "server"
    
    # Testing verb inflection stripping
    assert tokenizer.stem_word("running") == "runn"
    assert tokenizer.stem_word("tested") == "test"


def test_process_full_pipeline(tokenizer):
    input_text = "Developers are developing scalable search solutions!"
    processed = tokenizer.process(input_text)
    
    # Assert meaningful terms remain and stop words are filtered
    assert len(processed) > 0
    assert "are" not in processed