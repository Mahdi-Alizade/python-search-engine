import pytest
from fastapi.testclient import TestClient
from src.api import app, index_engine


@pytest.fixture
def client():
    # Setup fresh document in shared index
    index_engine.documents.clear()
    index_engine.index.clear()
    index_engine.add_document(
        doc_id="api_doc_1",
        title="Distributed Systems with Python",
        content="Building high throughput distributed search engines using FastAPI and inverted indexes.",
        url="https://example.com/api-doc"
    )

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_stats_endpoint(client):
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_documents"] == 1
    assert data["unique_terms"] > 0


def test_search_endpoint_found(client):
    response = client.get("/api/v1/search?q=FastAPI")
    assert response.status_code == 200
    data = response.json()
    assert data["total_found"] == 1
    assert data["results"][0]["doc_id"] == "api_doc_1"


def test_ingest_new_document_via_api(client):
    new_doc = {
        "doc_id": "api_doc_2",
        "title": "Redis In-Memory Store",
        "content": "Redis provides high performance key-value caching.",
        "url": "https://redis.io"
    }
    response = client.post("/api/v1/documents", json=new_doc)
    assert response.status_code == 201

    # Verify search finds the new document immediately
    search_res = client.get("/api/v1/search?q=Redis")
    assert search_res.status_code == 200
    assert search_res.json()["total_found"] >= 1