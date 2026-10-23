import math
from typing import Dict, List, Optional
from src.inverted_index import InvertedIndex


class BM25Scorer:
    """
    Computes Okapi BM25 relevance scores for document matching.
    Includes saturation parameter k1 and length normalization parameter b.
    """

    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def compute_average_doc_length(self) -> float:
        """Calculate the average token length across all indexed documents."""
        total_docs = self.index.total_documents
        if total_docs == 0:
            return 0.0

        accumulated_tokens = 0
        for doc_id, doc_record in self.index.documents.items():
            accumulated_tokens = accumulated_tokens + doc_record.total_tokens

        average_length = accumulated_tokens / total_docs
        return average_length

    def calculate_bm25_idf(self, term: str) -> float:
        """Calculate probabilistic Robertson-Spärck Jones IDF for a term."""
        postings = self.index.get_postings(term)
        doc_count_with_term = len(postings)
        total_docs = self.index.total_documents

        if doc_count_with_term == 0 or total_docs == 0:
            return 0.0

        # Standard BM25 IDF formulation with floor smoothing to prevent negative scores
        numerator = (total_docs - doc_count_with_term + 0.5)
        denominator = (doc_count_with_term + 0.5)
        raw_ratio = numerator / denominator

        if raw_ratio <= 0.0:
            return 0.0

        idf_score = math.log(raw_ratio + 1.0)
        return idf_score

    def score_document(self, term: str, doc_id: str, avg_doc_len: float) -> float:
        """Calculate the BM25 score contribution of a single term for a given document."""
        postings = self.index.get_postings(term)
        if doc_id not in postings:
            return 0.0

        doc_record = self.index.get_document(doc_id)
        if not doc_record or avg_doc_len == 0.0:
            return 0.0

        posting = postings[doc_id]
        term_frequency = posting.term_frequency
        doc_len = doc_record.total_tokens

        term_idf = self.calculate_bm25_idf(term)

        # Length normalization factor
        length_penalty = 1.0 - self.b + (self.b * (doc_len / avg_doc_len))

        # Saturation calculation
        freq_numerator = term_frequency * (self.k1 + 1.0)
        freq_denominator = term_frequency + (self.k1 * length_penalty)

        bm25_term_score = term_idf * (freq_numerator / freq_denominator)
        return bm25_term_score

    def score_query(self, query_terms: List[str]) -> Dict[str, float]:
        """Compute aggregated BM25 scores for all matching documents across query terms."""
        if not query_terms or self.index.total_documents == 0:
            return {}

        avg_length = self.compute_average_doc_length()
        aggregated_scores: Dict[str, float] = {}

        for term in query_terms:
            postings = self.index.get_postings(term)
            if not postings:
                continue

            for doc_id in postings.keys():
                term_score = self.score_document(term, doc_id, avg_length)
                existing_score = aggregated_scores.get(doc_id, 0.0)
                aggregated_scores[doc_id] = existing_score + term_score

        return aggregated_scores