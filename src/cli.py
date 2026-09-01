import os
import sys
from dotenv import load_dotenv
from src.inverted_index import InvertedIndex
from src.search_engine import SearchEngine
from src.document_ingestion import DocumentIngestion

load_dotenv()

DEFAULT_STORAGE = os.getenv("INDEX_STORAGE_PATH", "storage/inverted_index.json")
DEFAULT_DATA_DIR = os.getenv("DOCUMENTS_DATA_DIR", "data/documents")


def print_banner() -> None:
    print("=" * 60)
    print("       🔍 PYTHON SEARCH ENGINE CLI (TF-IDF RANKED)       ")
    print("=" * 60)


def run_cli() -> None:
    print_banner()
    index = InvertedIndex()
    ingestion = DocumentIngestion(index=index)

    # Automatically load existing index if present
    if os.path.exists(DEFAULT_STORAGE):
        print(f"[*] Found existing index at '{DEFAULT_STORAGE}'. Loading...")
        try:
            index.load_from_file(DEFAULT_STORAGE)
            print(f"[+] Loaded {index.total_documents} documents successfully.\n")
        except Exception as e:
            print(f"[!] Warning: Failed to load index: {e}\n")

    engine = SearchEngine(index=index)

    while True:
        print("\nCommands:")
        print("  1. search <query>       - Search ranked documents")
        print("  2. ingest <path>        - Ingest file or folder (default: data/documents)")
        print("  3. save [filepath]      - Save index to disk")
        print("  4. load [filepath]      - Load index from disk")
        print("  5. stats                - View index statistics")
        print("  6. exit                 - Exit application")

        choice = input("\n[SearchEngine] > ").strip()
        if not choice:
            continue

        parts = choice.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("6", "exit", "quit"):
            print("👋 Exiting Search Engine. Goodbye!")
            sys.exit(0)

        elif cmd in ("1", "search"):
            if not arg:
                query = input("Enter search query: ").strip()
            else:
                query = arg

            if not query:
                print("[!] Empty query. Please provide search terms.")
                continue

            results = engine.search(query, top_k=5)
            if not results:
                print(f"[-] No documents found matching '{query}'.")
            else:
                print(f"\nFound {len(results)} matching results for '{query}':")
                print("-" * 60)
                for idx, r in enumerate(results, 1):
                    print(f"[{idx}] {r.title} (Score: {r.score})")
                    if r.url:
                        print(f"    URL: {r.url}")
                    print(f"    Snippet: {r.snippet}")
                    print("-" * 60)

        elif cmd in ("2", "ingest"):
            target_path = arg.strip() or DEFAULT_DATA_DIR
            if not os.path.exists(target_path):
                print(f"[!] Path '{target_path}' does not exist.")
                continue

            if os.path.isfile(target_path):
                if target_path.endswith(".txt"):
                    ok = ingestion.ingest_text_file(target_path)
                    print(f"[+] Ingested text file: {ok}")
                elif target_path.endswith(".json"):
                    cnt = ingestion.ingest_json_file(target_path)
                    print(f"[+] Ingested {cnt} documents from JSON.")
            else:
                cnt = ingestion.ingest_directory(target_path)
                print(f"[+] Ingested {cnt} total documents from directory.")

        elif cmd in ("3", "save"):
            save_path = arg.strip() or DEFAULT_STORAGE
            index.save_to_file(save_path)
            print(f"[+] Index saved to '{save_path}'.")

        elif cmd in ("4", "load"):
            load_path = arg.strip() or DEFAULT_STORAGE
            try:
                index.load_from_file(load_path)
                print(f"[+] Index loaded successfully ({index.total_documents} documents).")
            except Exception as e:
                print(f"[!] Error loading index: {e}")

        elif cmd in ("5", "stats"):
            print(f"\n--- Index Statistics ---")
            print(f"Total Documents: {index.total_documents}")
            print(f"Unique Terms:    {len(index.index)}")
            print(f"------------------------")

        else:
            print(f"[!] Unknown command: '{cmd}'. Type a valid option.")


if __name__ == "__main__":
    run_cli()