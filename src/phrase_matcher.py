from typing import List, Set, Dict, Optional
from src.inverted_index import InvertedIndex
from src.tokenizer import TextTokenizer


class PhraseMatcher:
    """
    Evaluates exact phrase matching across indexed documents
    using token positions in the inverted index.
    """

    def __init__(self, index: InvertedIndex, tokenizer: Optional[TextTokenizer] = None):
        self.index = index
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = TextTokenizer()

    def find_phrase_matches(self, phrase_text: str) -> List[str]:
        """
        Return a list of doc_ids where the tokens appear strictly sequentially.
        """
        tokens = self.tokenizer.process(phrase_text)
        if not tokens:
            return []

        if len(tokens) == 1:
            single_postings = self.index.get_postings(tokens[0])
            matched_ids = list(single_postings.keys())
            return matched_ids

        # Step 1: Find candidate documents containing ALL required tokens
        candidate_docs: Optional[Set[str]] = None
        term_postings_map = {}

        for term in tokens:
            postings = self.index.get_postings(term)
            if not postings:
                return []
            
            term_postings_map[term] = postings
            doc_keys = set(postings.keys())

            if candidate_docs is None:
                candidate_docs = doc_keys
            else:
                candidate_docs = candidate_docs.intersection(doc_keys)

        if not candidate_docs:
            return []

        # Step 2: Verify strict sequential positions
        verified_doc_ids: List[str] = []
        first_term = tokens[0]

        for doc_id in candidate_docs:
            first_term_positions = term_postings_map[first_term][doc_id].positions

            # Check if any start position forms the complete sequential phrase
            is_matched_in_doc = False
            for start_pos in first_term_positions:
                sequence_intact = True

                for offset, next_term in enumerate(tokens[1:], start=1):
                    expected_pos = start_pos + offset
                    next_term_positions = term_postings_map[next_term][doc_id].positions

                    if expected_pos not in next_term_positions:
                        sequence_intact = False
                        break

                if sequence_intact:
                    is_matched_in_doc = True
                    break

            if is_matched_in_doc:
                verified_doc_ids.append(doc_id)

        return verified_doc_ids