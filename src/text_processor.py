import re
from typing import List, Set


class TextProcessor:
    """Handles text normalization, tokenization, stopword removal, and stemming."""

    DEFAULT_STOPWORDS: Set[str] = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves"
    }

    def __init__(self, stopwords: Set[str] = None, enable_stemming: bool = True):
        self.stopwords = stopwords if stopwords is not None else self.DEFAULT_STOPWORDS
        self.enable_stemming = enable_stemming
        self._token_pattern = re.compile(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b")

    def tokenize(self, text: str) -> List[str]:
        """Extract raw word tokens from text using regex matching."""
        if not text:
            return []
        return self._token_pattern.findall(text.lower())

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filter out stopwords and single-character noise tokens."""
        return [token for token in tokens if token not in self.stopwords and len(token) > 1]

    def stem(self, word: str) -> str:
        """Apply Porter-style suffix stripping heuristics for English terms."""
        if len(word) <= 3:
            return word

        # Step 1a: Plural forms handling
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2]
        elif word.endswith("ss"):
            pass
        elif any(word.endswith(sfx) for sfx in ["xes", "ses", "zes", "ches", "shes"]):
            word = word[:-2]
        elif word.endswith("s"):
            word = word[:-1]

        # Step 1b: Verb suffixes
        if word.endswith("eed"):
            if len(word) > 4:
                word = word[:-1]
        elif word.endswith("ed") and len(word) > 4:
            word = word[:-2]
        elif word.endswith("ing") and len(word) > 5:
            word = word[:-3]

        # Step 2: Suffix replacements
        suffix_replacements = [
            ("ational", "ate"),
            ("tional", "tion"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("ement", ""),
            ("ment", ""),
            ("able", ""),
            ("ible", ""),
            ("ance", ""),
            ("ence", ""),
            ("izer", "ize"),
            ("ator", "ate"),
            ("al", ""),
            ("ful", ""),
            ("ness", ""),
            ("ly", "")
        ]

        for suffix, replacement in suffix_replacements:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[:-len(suffix)] + replacement
                break

        return word

    def process(self, text: str) -> List[str]:
        """Execute the full text processing pipeline."""
        raw_tokens = self.tokenize(text)
        filtered_tokens = self.remove_stopwords(raw_tokens)

        if not self.enable_stemming:
            return filtered_tokens

        return [self.stem(token) for token in filtered_tokens]