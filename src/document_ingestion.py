import json
import os
from typing import List, Dict, Any, Optional
from src.inverted_index import InvertedIndex


class DocumentIngestion:
    """Handles loading documents from directories containing text and JSON files."""

    def __init__(self, index: InvertedIndex):
        self.index = index

    def ingest_text_file(self, file_path: str, doc_id: Optional[str] = None) -> bool:
        """Ingest a single .txt file using its filename as default doc_id and title."""
        if not os.path.isfile(file_path):
            return False

        filename = os.path.basename(file_path)
        derived_id = doc_id or os.path.splitext(filename)[0]
        title = derived_id.replace("_", " ").replace("-", " ").title()

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        self.index.add_document(doc_id=derived_id, title=title, content=content)
        return True

    def ingest_json_file(self, file_path: str) -> int:
        """Ingest documents from a .json file containing a single doc or a list of docs."""
        if not os.path.isfile(file_path):
            return 0

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        records: List[Dict[str, Any]] = data if isinstance(data, list) else [data]
        ingested_count = 0

        for item in records:
            if not isinstance(item, dict) or "doc_id" not in item or "content" not in item:
                continue

            self.index.add_document(
                doc_id=str(item["doc_id"]),
                title=str(item.get("title", f"Document {item['doc_id']}")),
                content=str(item["content"]),
                url=item.get("url")
            )
            ingested_count += 1

        return ingested_count

    def ingest_directory(self, directory_path: str) -> int:
        """Recursively scan a directory and ingest all .txt and .json documents."""
        if not os.path.isdir(directory_path):
            return 0

        total_ingested = 0
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                if ext == ".txt":
                    if self.ingest_text_file(file_path):
                        total_ingested += 1
                elif ext == ".json":
                    total_ingested += self.ingest_json_file(file_path)

        return total_ingested