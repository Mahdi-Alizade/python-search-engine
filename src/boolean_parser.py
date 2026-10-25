from typing import Set, List, Optional
from src.inverted_index import InvertedIndex
from src.tokenizer import TextTokenizer


class BooleanQueryEngine:
    """
    Parses and evaluates queries with boolean operators (AND, OR, NOT).
    Supports token matching against inverted index postings.
    """

    def __init__(self, index: InvertedIndex, tokenizer: Optional[TextTokenizer] = None):
        self.index = index
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = TextTokenizer()

    def get_all_document_ids(self) -> Set[str]:
        """Fetch all indexed document identifiers in the corpus."""
        all_ids = set(self.index.documents.keys())
        return all_ids

    def _get_term_docs(self, term: str) -> Set[str]:
        """Retrieve the set of doc_ids containing the processed term."""
        processed_tokens = self.tokenizer.process(term)
        if not processed_tokens:
            return set()

        first_term = processed_tokens[0]
        postings = self.index.get_postings(first_term)
        matched_doc_ids = set(postings.keys())
        return matched_doc_ids

    def evaluate(self, query: str) -> Set[str]:
        """
        Evaluate boolean expression. Supported syntax examples:
          - python AND backend
          - python OR redis
          - python NOT database
        """
        clean_query = query.strip()
        if not clean_query:
            return set()

        raw_parts = clean_query.split()
        if not raw_parts:
            return set()

        # Step 1: Initial accumulator setup with the first operand
        first_token = raw_parts[0]
        if first_token.upper() == "NOT" and len(raw_parts) > 1:
            excluded_docs = self._get_term_docs(raw_parts[1])
            active_docs = self.get_all_document_ids() - excluded_docs
            cursor = 2
        else:
            active_docs = self._get_term_docs(first_token)
            cursor = 1

        # Step 2: Sequentially process operator-operand pairs
        while cursor < len(raw_parts):
            operator = raw_parts[cursor].upper()

            if cursor + 1 >= len(raw_parts):
                break

            target_term = raw_parts[cursor + 1]
            target_docs = self._get_term_docs(target_term)

            if operator == "AND":
                active_docs = active_docs.intersection(target_docs)
            elif operator == "OR":
                active_docs = active_docs.union(target_docs)
            elif operator == "NOT":
                active_docs = active_docs.difference(target_docs)
            else:
                # Default fallback behavior: treat space as OR
                active_docs = active_docs.union(self._get_term_docs(raw_parts[cursor]))
                cursor = cursor + 1
                continue

            cursor = cursor + 2

        return active_docs