import re
from dataclasses import dataclass
from typing import List, Optional, Set
from src.inverted_index import InvertedIndex
from src.tokenizer import TextTokenizer
from src.bm25_scorer import BM25Scorer
from src.phrase_matcher import PhraseMatcher


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
    Unified search engine supporting Okapi BM25 relevance scoring
    and exact sequential phrase matching.
    """

    def __init__(
        self,
        index: Optional[InvertedIndex] = None,
        tokenizer: Optional[TextTokenizer] = None,
        k1: float = 1.5,
        b: float = 0.75
    ):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = TextTokenizer()

        if index is not None:
            self.index = index
        else:
            self.index = InvertedIndex(tokenizer=self.tokenizer)

        self.bm25 = BM25Scorer(index=self.index, k1=k1, b=b)
        self.phrase_matcher = PhraseMatcher(index=self.index, tokenizer=self.tokenizer)
        self.phrase_pattern = re.compile(r'"([^"]+)"')

    def _extract_phrases_and_terms(self, query: str):
        """Extract quoted phrases and residual unquoted terms from query string."""
        raw_phrases = self.phrase_pattern.findall(query)
        cleaned_phrases = []
        for p in raw_phrases:
            stripped = p.strip()
            if stripped:
                cleaned_phrases.append(stripped)

        # Remove quoted parts to leave free terms
        unquoted_query = self.phrase_pattern.sub(" ", query)
        free_terms = self.tokenizer.process(unquoted_query)

        return cleaned_phrases, free_terms

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
        """
        Execute unified search. Applies phrase constraint filtering
        and scores candidates using Okapi BM25.
        """
        stripped_query = query.strip()
        if not stripped_query:
            return []

        if self.index.total_documents == 0:
            return []

        phrases, free_terms = self._extract_phrases_and_terms(stripped_query)

        # Collect all query terms for scoring and snippet generation
        all_scoring_terms = list(free_terms)
        for phrase in phrases:
            phrase_tokens = self.tokenizer.process(phrase)
            all_scoring_terms.extend(phrase_tokens)

        if not all_scoring_terms:
            return []

        # Step 1: Calculate baseline BM25 scores
        doc_scores = self.bm25.score_query(all_scoring_terms)
        if not doc_scores:
            return []

        candidate_ids: Set[str] = set(doc_scores.keys())

        # Step 2: Enforce exact phrase match constraint if phrases exist
        if phrases:
            for phrase in phrases:
                matching_phrase_docs = set(self.phrase_matcher.find_phrase_matches(phrase))
                candidate_ids = candidate_ids.intersection(matching_phrase_docs)

        if not candidate_ids:
            return []

        # Step 3: Sort filtered candidates descending by BM25 score
        sorted_doc_ids = sorted(
            candidate_ids,
            key=lambda d_id: doc_scores[d_id],
            reverse=True
        )

        results = []
        for doc_id in sorted_doc_ids[:top_k]:
            doc_record = self.index.get_document(doc_id)
            if not doc_record:
                continue

            rounded_score = round(doc_scores[doc_id], 4)
            snippet = self._extract_snippet(doc_record.content, all_scoring_terms)

            result_item = SearchResult(
                doc_id=doc_record.doc_id,
                title=doc_record.title,
                content=doc_record.content,
                url=doc_record.url,
                score=rounded_score,
                snippet=snippet
            )
            results.append(result_item)

        return results