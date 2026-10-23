import os
import json
from typing import Dict, List, Optional, Any
from src.models import DocumentRecord, TermPosting
from src.tokenizer import TextTokenizer


class InvertedIndex:
    """
    Core Inverted Index engine maintaining term-to-document postings
    and positional occurrence mappings.
    """

    def __init__(self, tokenizer: Optional[TextTokenizer] = None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = TextTokenizer()

        # Storage mapping: term -> {doc_id: TermPosting}
        self.index: Dict[str, Dict[str, TermPosting]] = {}

        # Document registry: doc_id -> DocumentRecord
        self.documents: Dict[str, DocumentRecord] = {}

    @property
    def total_documents(self) -> int:
        """Return the total number of indexed documents."""
        count = len(self.documents)
        return count

    def add_document(self, doc_id: str, title: str, content: str, url: Optional[str] = None) -> None:
        """Tokenize document text, track positions, and register into the inverted index."""
        combined_text = f"{title} {content}"
        tokens = self.tokenizer.process(combined_text)
        token_count = len(tokens)

        # Register document metadata
        doc_record = DocumentRecord(
            doc_id=doc_id,
            title=title,
            content=content,
            url=url,
            total_tokens=token_count
        )
        self.documents[doc_id] = doc_record

        # Track occurrence positions of each token in this document
        position_map: Dict[str, List[int]] = {}
        for current_pos, term in enumerate(tokens):
            if term not in position_map:
                position_map[term] = []
            position_map[term].append(current_pos)

        # Update global postings list
        for term, occurrences in position_map.items():
            if term not in self.index:
                self.index[term] = {}

            posting = TermPosting(
                doc_id=doc_id,
                term_frequency=len(occurrences),
                positions=occurrences
            )
            self.index[term][doc_id] = posting

    def get_postings(self, term: str) -> Dict[str, TermPosting]:
        """Retrieve postings dictionary for a specific processed term."""
        processed_terms = self.tokenizer.process(term)
        if not processed_terms:
            return {}

        lookup_key = processed_terms[0]
        if lookup_key in self.index:
            return self.index[lookup_key]
        return {}

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Fetch document record by unique document identifier."""
        if doc_id in self.documents:
            return self.documents[doc_id]
        return None

    def save_to_disk(self, target_filepath: str) -> None:
        """Serialize index and document store into a JSON file."""
        parent_dir = os.path.dirname(os.path.abspath(target_filepath))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        serialized_index: Dict[str, Dict[str, Any]] = {}
        for term, doc_map in self.index.items():
            serialized_index[term] = {}
            for doc_id, posting in doc_map.items():
                serialized_index[term][doc_id] = posting.to_dict()

        serialized_documents: Dict[str, Any] = {}
        for doc_id, doc_record in self.documents.items():
            serialized_documents[doc_id] = doc_record.to_dict()

        payload = {
            "schema_version": "1.0",
            "total_documents": self.total_documents,
            "documents": serialized_documents,
            "index": serialized_index
        }

        with open(target_filepath, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)

    def load_from_disk(self, target_filepath: str) -> None:
        """Load index mapping and document metadata from a JSON file."""
        if not os.path.exists(target_filepath):
            raise FileNotFoundError(f"Storage file does not exist: {target_filepath}")

        with open(target_filepath, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        # Restore documents registry
        raw_docs = payload.get("documents", {})
        self.documents = {}
        for doc_id, raw_doc_data in raw_docs.items():
            self.documents[doc_id] = DocumentRecord.from_dict(raw_doc_data)

        # Restore postings index
        raw_index = payload.get("index", {})
        self.index = {}
        for term, postings_map in raw_index.items():
            self.index[term] = {}
            for doc_id, posting_data in postings_map.items():
                self.index[term][doc_id] = TermPosting.from_dict(posting_data)