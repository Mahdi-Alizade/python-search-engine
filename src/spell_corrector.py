from typing import List, Optional, Set
from src.inverted_index import InvertedIndex
from src.tokenizer import TextTokenizer


class SpellCorrector:
    """
    Provides spelling corrections and 'Did you mean' suggestions
    based on indexed corpus vocabulary and Levenshtein edit distance.
    """

    def __init__(self, index: InvertedIndex, tokenizer: Optional[TextTokenizer] = None):
        self.index = index
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = TextTokenizer()

    def get_vocabulary(self) -> Set[str]:
        """Extract all indexed unique terms currently in the inverted index."""
        unique_terms = set(self.index.index.keys())
        return unique_terms

    def calculate_levenshtein_distance(self, source_word: str, target_word: str) -> int:
        """
        Compute standard Levenshtein edit distance between two strings
        using an iterative matrix formulation.
        """
        source_len = len(source_word)
        target_len = len(target_word)

        if source_len == 0:
            return target_len
        if target_len == 0:
            return source_len

        # Matrix initialization: (source_len + 1) x (target_len + 1)
        matrix = []
        for i in range(source_len + 1):
            row = [0] * (target_len + 1)
            matrix.append(row)

        for i in range(source_len + 1):
            matrix[i][0] = i
        for j in range(target_len + 1):
            matrix[0][j] = j

        for i in range(1, source_len + 1):
            source_char = source_word[i - 1]
            for j in range(1, target_len + 1):
                target_char = target_word[j - 1]

                if source_char == target_char:
                    cost = 0
                else:
                    cost = 1

                deletion_cost = matrix[i - 1][j] + 1
                insertion_cost = matrix[i][j - 1] + 1
                substitution_cost = matrix[i - 1][j - 1] + cost

                matrix[i][j] = min(deletion_cost, insertion_cost, substitution_cost)

        final_distance = matrix[source_len][target_len]
        return final_distance

    def suggest_term(self, input_word: str, max_distance: int = 2) -> Optional[str]:
        """
        Find the closest vocabulary term within max_distance.
        Breaks ties by choosing terms with higher document frequency.
        """
        processed_tokens = self.tokenizer.process(input_word)
        if not processed_tokens:
            return None

        lookup_term = processed_tokens[0]
        vocab = self.get_vocabulary()

        # If term already exists perfectly in index, no correction is required
        if lookup_term in vocab:
            return lookup_term

        best_suggestion = None
        lowest_distance = max_distance + 1
        highest_doc_frequency = -1

        for candidate in vocab:
            dist = self.calculate_levenshtein_distance(lookup_term, candidate)
            if dist <= max_distance:
                candidate_postings = self.index.get_postings(candidate)
                doc_freq = len(candidate_postings)

                if dist < lowest_distance:
                    lowest_distance = dist
                    best_suggestion = candidate
                    highest_doc_frequency = doc_freq
                elif dist == lowest_distance:
                    if doc_freq > highest_doc_frequency:
                        best_suggestion = candidate
                        highest_doc_frequency = doc_freq

        return best_suggestion

    def suggest_query(self, query: str) -> Optional[str]:
        """
        Suggest corrected full query if any misspelled words are detected.
        Returns corrected string, or None if query needs no adjustment.
        """
        raw_words = query.split()
        if not raw_words:
            return None

        has_correction = False
        suggested_words: List[str] = []

        for word in raw_words:
            clean_word = word.strip(".,!?:;\"'()[]{}")
            if not clean_word:
                suggested_words.append(word)
                continue

            # Skip checking uppercase boolean operators
            if clean_word in {"AND", "OR", "NOT"}:
                suggested_words.append(clean_word)
                continue

            suggested = self.suggest_term(clean_word)
            if suggested is not None and suggested != clean_word.lower():
                suggested_words.append(suggested)
                has_correction = True
            else:
                suggested_words.append(clean_word)

        if has_correction:
            return " ".join(suggested_words)

        return None