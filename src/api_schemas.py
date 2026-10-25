from typing import List, Optional
from pydantic import BaseModel, Field


class IngestDocumentRequest(BaseModel):
    """Payload for indexing a single document via HTTP."""
    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document headline or title")
    content: str = Field(..., description="Full text body to index")
    url: Optional[str] = Field(None, description="Optional canonical URL link")


class SearchResultItem(BaseModel):
    """Output structure for a single matched document."""
    doc_id: str
    title: str
    url: Optional[str]
    score: float
    snippet: str


class SearchQueryResponse(BaseModel):
    """Unified response containing ranked matches and optional typo suggestions."""
    query: str
    total_found: int
    did_you_mean: Optional[str] = None
    results: List[SearchResultItem]


class IndexStatsResponse(BaseModel):
    """Metrics reflecting current index status."""
    total_documents: int
    unique_terms: int