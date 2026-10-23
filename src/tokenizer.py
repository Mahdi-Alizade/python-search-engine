import re
from typing import List, Set


# Common English stopwords list kept locally without heavy dependencies
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


class TextTokenizer:
    """
    Handles normalization, tokenization, stopword removal,
    and term stemming for document indexing and searching.
    """

    def __init__(self, custom_stopwords: Set[str] = None, enable_stem: bool = True):
        if custom_stopwords is not None:
            self.stopwords = custom_stopwords
        else:
            self.stopwords = DEFAULT_STOPWORDS

        self.enable_stem = enable_stem
        self.word_pattern = re.compile(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b")

    def normalize(self, text: str) -> str:
        """Convert input text to lowercase and strip whitespace."""
        if not text:
            return ""
        lowered_text = text.lower()
        cleaned_text = lowered_text.strip()
        return cleaned_text

    def extract_tokens(self, text: str) -> List[str]:
        """Extract alphanumeric tokens matching word boundaries."""
        normalized = self.normalize(text)
        if not normalized:
            return []
        tokens = self.word_pattern.findall(normalized)
        return tokens

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Remove common stopwords and single character noise."""
        result = []
        for word in tokens:
            is_stopword = word in self.stopwords
            is_single_letter = len(word) <= 1
            if not is_stopword and not is_single_letter:
                result.append(word)
        return result

    def stem_word(self, word: str) -> str:
        """Apply simple rule-based suffix reduction for English words."""
        target = word
        if len(target) <= 3:
            return target

        # Step 1: Plurals and past tense
        if target.endswith("sses"):
            target = target[:-2]
        elif target.endswith("ies"):
            target = target[:-2]
        elif target.endswith("ss"):
            pass
        elif any(target.endswith(end) for end in ["xes", "ses", "zes", "ches", "shes"]):
            target = target[:-2]
        elif target.endswith("s"):
            target = target[:-1]

        # Step 2: Verb endings
        if target.endswith("eed"):
            if len(target) > 4:
                target = target[:-1]
        elif target.endswith("ed") and len(target) > 4:
            target = target[:-2]
        elif target.endswith("ing") and len(target) > 5:
            target = target[:-3]

        # Step 3: Derivational suffixes
        suffix_rules = [
            ("ation", "ate"),
            ("ization", "ize"),
            ("tional", "tion"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("ement", ""),
            ("able", ""),
            ("ible", ""),
            ("ance", ""),
            ("ence", ""),
            ("izer", "ize"),
            ("ator", "ate"),
            ("ness", ""),
            ("ful", ""),
            ("ly", "")
        ]

        for suffix, replacement in suffix_rules:
            if target.endswith(suffix):
                prefix_len = len(target) - len(suffix)
                if prefix_len >= 3:
                    target = target[:-len(suffix)] + replacement
                    break

        return target

    def process(self, text: str) -> List[str]:
        """Run full preprocessing pipeline on raw text input."""
        tokens = self.extract_tokens(text)
        filtered = self.filter_tokens(tokens)

        if not self.enable_stem:
            return filtered

        stemmed_tokens = []
        for item in filtered:
            stemmed_word = self.stem_word(item)
            stemmed_tokens.append(stemmed_word)

        return stemmed_tokens