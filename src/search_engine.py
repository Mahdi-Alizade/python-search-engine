import math
from dataclasses import dataclass
from typing import List, Optional
from src.inverted_index import InvertedIndex
from src.models import DocumentRecord
from src.tokenizer import TextTokenizer


@dataclass
class SearchResult:
    """Represents a matched and scored search result item."""
    doc_id: str
    title: str
    content: str
    url: Optional[str]
    score: float
    snippet: str


class SearchEngine:
    """
    Search engine ranking documents using Term Frequency - Inverse Document Frequency (TF-IDF).
    """

    def __init__(self, index: Optional[InvertedIndex] = None, tokenizer: Optional[TextTokenizer] = None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = TextTokenizer()

        if index is not None:
            self.index = index
        else:
            self.index = InvertedIndex(tokenizer=self.tokenizer)

    def _calculate_smooth_idf(self, term: str) -> float:
        """Calculate smoothed inverse document frequency."""
        postings = self.index.get_postings(term)
        doc_frequency = len(postings)
        if doc_frequency == 0:
            return 0.0

        total_docs = self.index.total_documents
        # Using smooth logarithmic IDF formulation
        idf_ratio = (total_docs / doc_frequency) + 1.0
        return math.log(idf_ratio)

    def _extract_snippet(self, raw_content: str, query_terms: List[str], max_chars: int = 140) -> str:
        """Construct a contextual text snippet around matched terms."""
        if not raw_content:
            return ""

        words = raw_content.split()
        if len(raw_content) <= max_chars:
            return raw_content

        lowered_words = [w.lower().strip(".,!?:;\"'()[]{}") for w in words]
        matched_positions = []
        for idx, word in enumerate(lowered_words):
            for term in query_terms:
                if term in word:
                    matched_positions.append(idx)
                    break

        if not matched_positions:
            fallback = " ".join(words[:16]) + "..."
            return fallback

        first_match = matched_positions[0]
        start_bound = max(0, first_match - 5)
        end_bound = min(len(words), first_match + 12)

        selected_words = words[start_bound:end_bound]
        snippet_text = " ".join(selected_words)

        if start_bound > 0:
            snippet_text = "..." + snippet_text
        if end_bound < len(words):
            snippet_text = snippet_text + "..."

        return snippet_text

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Rank matching documents based on TF-IDF relevance."""
        query_tokens = self.tokenizer.process(query)
        if not query_tokens:
            return []

        if self.index.total_documents == 0:
            return []

        doc_accumulated_scores = {}

        for term in query_tokens:
            postings_map = self.index.get_postings(term)
            if not postings_map:
                continue

            term_idf = self._calculate_smooth_idf(term)

            for doc_id, posting in postings_map.items():
                doc_record = self.index.get_document(doc_id)
                if not doc_record:
                    continue

                if doc_record.total_tokens == 0:
                    continue

                # Term frequency normalized by document token count
                tf = posting.term_frequency / doc_record.total_tokens
                score_contribution = tf * term_idf

                current_score = doc_accumulated_scores.get(doc_id, 0.0)
                doc_accumulated_scores[doc_id] = current_score + score_contribution

        if not doc_accumulated_scores:
            return []

        # Sort documents descending by score
        ranked_doc_ids = sorted(
            doc_accumulated_scores.keys(),
            key=lambda d_id: doc_accumulated_scores[d_id],
            reverse=True
        )

        results = []
        for doc_id in ranked_doc_ids[:top_k]:
            doc_record = self.index.get_document(doc_id)
            rounded_score = round(doc_accumulated_scores[doc_id], 4)
            snippet = self._extract_snippet(doc_record.content, query_tokens)

            item = SearchResult(
                doc_id=doc_record.doc_id,
                title=doc_record.title,
                content=doc_record.content,
                url=doc_record.url,
                score=rounded_score,
                snippet=snippet
            )
            results.append(item)

        return results