from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DocumentRecord:
    """Represents an ingested document with metadata and content."""
    doc_id: str
    title: str
    content: str
    url: Optional[str] = None
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert document instance to serializable dictionary."""
        data = {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "total_tokens": self.total_tokens
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentRecord":
        """Instantiate DocumentRecord from raw dictionary."""
        record = cls(
            doc_id=data.get("doc_id", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            url=data.get("url"),
            total_tokens=data.get("total_tokens", 0)
        )
        return record


@dataclass
class TermPosting:
    """Tracks frequency and positional occurrences of a term within a document."""
    doc_id: str
    term_frequency: int = 0
    positions: List[int] = field(default_factory=list)

    def add_occurrence(self, position: int) -> None:
        """Register a new occurrence position for the term."""
        self.positions.append(position)
        self.term_frequency = len(self.positions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert posting to dictionary for serialization."""
        data = {
            "doc_id": self.doc_id,
            "term_frequency": self.term_frequency,
            "positions": self.positions
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TermPosting":
        """Reconstruct TermPosting from dictionary payload."""
        posting = cls(
            doc_id=data.get("doc_id", ""),
            term_frequency=data.get("term_frequency", 0),
            positions=data.get("positions", [])
        )
        return posting