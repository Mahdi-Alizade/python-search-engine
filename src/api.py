import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Query, HTTPException, status
from dotenv import load_dotenv

from src.inverted_index import InvertedIndex
from src.search_engine import SearchEngine
from src.spell_corrector import SpellCorrector
from src.api_schemas import (
    IngestDocumentRequest,
    SearchQueryResponse,
    SearchResultItem,
    IndexStatsResponse
)

load_dotenv()

DEFAULT_STORAGE = os.getenv("INDEX_STORAGE_PATH", "storage/inverted_index.json")

# Shared in-memory engine instances
index_engine = InvertedIndex()
search_engine: Optional[SearchEngine] = None
spell_corrector: Optional[SpellCorrector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize index on startup and persist on shutdown."""
    global search_engine, spell_corrector

    if os.path.exists(DEFAULT_STORAGE):
        try:
            index_engine.load_from_disk(DEFAULT_STORAGE)
        except Exception:
            pass

    search_engine = SearchEngine(index=index_engine)
    spell_corrector = SpellCorrector(index=index_engine)

    yield

    # Auto-save index on shutdown
    if index_engine.total_documents > 0:
        index_engine.save_to_disk(DEFAULT_STORAGE)


app = FastAPI(
    title="Python Full-Text Search Engine API",
    description="REST API powered by Okapi BM25, exact phrase search, and spell correction.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", tags=["System"])
def health_check():
    """Health check ping endpoint."""
    return {"status": "healthy", "service": "search-engine-api"}


@app.get("/api/v1/stats", response_model=IndexStatsResponse, tags=["Index"])
def get_index_metrics():
    """Fetch current count of indexed documents and unique vocabulary size."""
    doc_count = index_engine.total_documents
    vocab_count = len(index_engine.index)
    return IndexStatsResponse(
        total_documents=doc_count,
        unique_terms=vocab_count
    )


@app.post("/api/v1/documents", status_code=status.HTTP_201_CREATED, tags=["Index"])
def ingest_document(payload: IngestDocumentRequest):
    """Index a single document into the active inverted index."""
    if not payload.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document content cannot be empty."
        )

    index_engine.add_document(
        doc_id=payload.doc_id,
        title=payload.title,
        content=payload.content,
        url=payload.url
    )

    return {
        "message": f"Document '{payload.doc_id}' indexed successfully.",
        "total_documents": index_engine.total_documents
    }


@app.get("/api/v1/search", response_model=SearchQueryResponse, tags=["Search"])
def execute_search(
    q: str = Query(..., min_length=1, description="Search query string"),
    limit: int = Query(10, ge=1, le=50, description="Max results to return")
):
    """
    Search indexed documents with BM25 ranking, exact quotes, and spell suggestions.
    """
    cleaned_query = q.strip()
    if not cleaned_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be blank."
        )

    # Compute spelling suggestion
    suggested_correction = None
    if spell_corrector is not None:
        suggested_correction = spell_corrector.suggest_query(cleaned_query)

    # Execute search
    matches = search_engine.search(cleaned_query, top_k=limit)

    # Fallback to corrected query if original returns no results
    if not matches and suggested_correction:
        matches = search_engine.search(suggested_correction, top_k=limit)

    result_items = []
    for item in matches:
        result_items.append(
            SearchResultItem(
                doc_id=item.doc_id,
                title=item.title,
                url=item.url,
                score=item.score,
                snippet=item.snippet
            )
        )

    return SearchQueryResponse(
        query=cleaned_query,
        total_found=len(result_items),
        did_you_mean=suggested_correction,
        results=result_items
    )