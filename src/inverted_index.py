import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from src.text_processor import TextProcessor


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    url: Optional[str] = None
    token_count: int = 0


@dataclass
class Posting:
    doc_id: str
    term_frequency: int
    positions: List[int]


class InvertedIndex:
    """Manages inverted index mapping, document registries, and disk persistence."""

    def __init__(self, text_processor: Optional[TextProcessor] = None):
        self.processor = text_processor or TextProcessor()
        # index mapping: term -> {doc_id: Posting}
        self.index: Dict[str, Dict[str, Posting]] = {}
        # documents registry: doc_id -> Document
        self.documents: Dict[str, Document] = {}

    @property
    def total_documents(self) -> int:
        return len(self.documents)

    def add_document(self, doc_id: str, title: str, content: str, url: Optional[str] = None) -> None:
        """Process and index a single document into the inverted index."""
        tokens = self.processor.process(f"{title} {content}")
        token_count = len(tokens)

        doc = Document(
            doc_id=doc_id,
            title=title,
            content=content,
            url=url,
            token_count=token_count
        )
        self.documents[doc_id] = doc

        # Count frequencies and track positions
        term_positions: Dict[str, List[int]] = {}
        for pos, token in enumerate(tokens):
            term_positions.setdefault(token, []).append(pos)

        for token, positions in term_positions.items():
            if token not in self.index:
                self.index[token] = {}
            self.index[token][doc_id] = Posting(
                doc_id=doc_id,
                term_frequency=len(positions),
                positions=positions
            )

    def get_postings(self, term: str) -> Dict[str, Posting]:
        """Retrieve postings list for a given term."""
        processed_tokens = self.processor.process(term)
        if not processed_tokens:
            return {}
        processed_term = processed_tokens[0]
        return self.index.get(processed_term, {})

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve document metadata by doc_id."""
        return self.documents.get(doc_id)

    def save_to_file(self, file_path: str) -> None:
        """Serialize index and documents to a JSON file on disk."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        serializable_index: Dict[str, Dict[str, Any]] = {}
        for term, postings in self.index.items():
            serializable_index[term] = {
                doc_id: asdict(posting) for doc_id, posting in postings.items()
            }

        serializable_docs = {
            doc_id: asdict(doc) for doc_id, doc in self.documents.items()
        }

        payload = {
            "version": "1.0",
            "total_documents": self.total_documents,
            "documents": serializable_docs,
            "index": serializable_index
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_from_file(self, file_path: str) -> None:
        """Load and deserialize index and documents from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Index file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.documents = {
            doc_id: Document(**doc_data)
            for doc_id, doc_data in payload.get("documents", {}).items()
        }

        self.index = {}
        for term, postings_map in payload.get("index", {}).items():
            self.index[term] = {
                doc_id: Posting(**posting_data)
                for doc_id, posting_data in postings_map.items()
            }