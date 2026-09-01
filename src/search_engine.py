import math
from dataclasses import dataclass
from typing import List, Optional
from src.inverted_index import InvertedIndex, Document
from src.text_processor import TextProcessor


@dataclass
class SearchResult:
    doc_id: str
    title: str
    content: str
    url: Optional[str]
    score: float
    snippet: str


class SearchEngine:
    """Core search engine responsible for querying, scoring with TF-IDF, and snippet generation."""

    def __init__(self, index: Optional[InvertedIndex] = None, processor: Optional[TextProcessor] = None):
        self.processor = processor or TextProcessor()
        self.index = index or InvertedIndex(text_processor=self.processor)

    def _calculate_idf(self, term: str) -> float:
        """Compute smooth Inverse Document Frequency for a term."""
        postings = self.index.get_postings(term)
        doc_frequency = len(postings)
        if doc_frequency == 0:
            return 0.0
        total_docs = self.index.total_documents
        return math.log((total_docs / doc_frequency) + 1.0)

    def _generate_snippet(self, content: str, query_tokens: List[str], max_length: int = 140) -> str:
        """Extract a contextual text snippet surrounding matched query terms."""
        if not content:
            return ""

        words = content.split()
        if len(content) <= max_length:
            return content

        lowered_words = [w.lower().strip(".,!?:;\"'()[]{}") for w in words]
        matched_indices = [
            idx for idx, word in enumerate(lowered_words)
            if any(token in word for token in query_tokens)
        ]

        if not matched_indices:
            return " ".join(words[:18]) + "..."

        target_idx = matched_indices[0]
        start = max(0, target_idx - 6)
        end = min(len(words), target_idx + 12)

        snippet = " ".join(words[start:end])
        if start > 0:
            snippet = "..." + snippet
        if end < len(words):
            snippet = snippet + "..."

        return snippet

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Execute query search, rank matching documents by TF-IDF, and return top results."""
        query_tokens = self.processor.process(query)
        if not query_tokens or self.index.total_documents == 0:
            return []

        doc_scores = {}

        for token in query_tokens:
            postings = self.index.get_postings(token)
            if not postings:
                continue

            idf = self._calculate_idf(token)

            for doc_id, posting in postings.items():
                doc = self.index.get_document(doc_id)
                if not doc or doc.token_count == 0:
                    continue

                # Normalized Term Frequency
                tf = posting.term_frequency / doc.token_count
                tf_idf = tf * idf

                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + tf_idf

        if not doc_scores:
            return []

        # Sort documents by descending score
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda d_id: doc_scores[d_id], reverse=True)

        results = []
        for doc_id in sorted_doc_ids[:top_k]:
            doc = self.index.get_document(doc_id)
            score = round(doc_scores[doc_id], 4)
            snippet = self._generate_snippet(doc.content, query_tokens)

            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    url=doc.url,
                    score=score,
                    snippet=snippet
                )
            )

        return results