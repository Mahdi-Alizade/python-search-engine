import json
import os
import pytest
from src.inverted_index import InvertedIndex
from src.document_ingestion import DocumentIngestion


@pytest.fixture
def index_and_ingestion():
    idx = InvertedIndex()
    ingestion = DocumentIngestion(index=idx)
    return idx, ingestion


def test_ingest_single_text_file(index_and_ingestion, tmp_path):
    idx, ingestion = index_and_ingestion
    file_path = os.path.join(tmp_path, "python-microservices.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Microservices architectures in Python with FastAPI and gRPC.")

    success = ingestion.ingest_text_file(file_path)
    assert success is True
    assert idx.total_documents == 1
    doc = idx.get_document("python-microservices")
    assert doc is not None
    assert "FastAPI" in doc.content


def test_ingest_json_file(index_and_ingestion, tmp_path):
    idx, ingestion = index_and_ingestion
    file_path = os.path.join(tmp_path, "docs.json")

    sample_docs = [
        {"doc_id": "api_1", "title": "API Gateway", "content": "Reverse proxy routing.", "url": "https://api.dev"},
        {"doc_id": "db_1", "title": "NoSQL DB", "content": "Key-value store performance.", "url": "https://db.dev"}
    ]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sample_docs, f)

    count = ingestion.ingest_json_file(file_path)
    assert count == 2
    assert idx.total_documents == 2
    assert idx.get_document("api_1").title == "API Gateway"


def test_ingest_directory_mixed_files(index_and_ingestion, tmp_path):
    idx, ingestion = index_and_ingestion

    # Write a TXT file
    with open(os.path.join(tmp_path, "doc1.txt"), "w", encoding="utf-8") as f:
        f.write("Clean architecture and unit testing.")

    # Write a JSON file
    with open(os.path.join(tmp_path, "doc2.json"), "w", encoding="utf-8") as f:
        json.dump({"doc_id": "doc2", "title": "Docker Containers", "content": "Containerizing backend apps."}, f)

    count = ingestion.ingest_directory(str(tmp_path))
    assert count == 2
    assert idx.total_documents == 2