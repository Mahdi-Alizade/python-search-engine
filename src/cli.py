import os
import sys
from dotenv import load_dotenv
from src.inverted_index import InvertedIndex
from src.search_engine import SearchEngine
from src.document_ingestion import DocumentIngestion
from src.spell_corrector import SpellCorrector

load_dotenv()

DEFAULT_STORAGE_FILE = os.getenv("INDEX_STORAGE_PATH", "storage/inverted_index.json")
DEFAULT_DATA_FOLDER = os.getenv("DOCUMENTS_DATA_DIR", "data/documents")


def display_welcome_banner() -> None:
    print("=" * 65)
    print("      🔍 PYTHON SEARCH ENGINE CLI (BM25 + SPELL CHECK)      ")
    print("=" * 65)


def run_cli() -> None:
    display_welcome_banner()
    index_engine = InvertedIndex()
    ingestion_service = DocumentIngestion(index=index_engine)

    # Attempt to load persistent index if available on disk
    if os.path.exists(DEFAULT_STORAGE_FILE):
        print(f"[*] Found serialized index at '{DEFAULT_STORAGE_FILE}'. Restoring...")
        try:
            index_engine.load_from_disk(DEFAULT_STORAGE_FILE)
            print(f"[+] Restored {index_engine.total_documents} documents successfully.\n")
        except Exception as load_err:
            print(f"[!] Warning: Could not restore index: {load_err}\n")

    search_core = SearchEngine(index=index_engine)
    spell_checker = SpellCorrector(index=index_engine)

    while True:
        print("\nAvailable Commands:")
        print("  1. search <query>       - Search documents (BM25, quotes for phrases, AND/OR/NOT)")
        print("  2. ingest <path>        - Ingest file or folder (default: data/documents)")
        print("  3. save [filepath]      - Persist index to JSON")
        print("  4. load [filepath]      - Load index from JSON")
        print("  5. stats                - View corpus and vocabulary metrics")
        print("  6. exit                 - Exit interactive session")

        try:
            user_input = input("\n[SearchEngine] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Terminating session. Goodbye!")
            sys.exit(0)

        if not user_input:
            continue

        command_segments = user_input.split(maxsplit=1)
        action = command_segments[0].lower()
        parameter = command_segments[1] if len(command_segments) > 1 else ""

        if action in ("6", "exit", "quit"):
            print("👋 Exiting Search Engine. Goodbye!")
            sys.exit(0)

        elif action in ("1", "search"):
            if not parameter:
                query_text = input("Enter search query: ").strip()
            else:
                query_text = parameter

            if not query_text:
                print("[!] Query cannot be empty.")
                continue

            # Check for possible spelling suggestions
            suggested_query = spell_checker.suggest_query(query_text)
            if suggested_query:
                print(f"\n💡 Did you mean: \033[93m{suggested_query}\033[0m")

            results = search_core.search(query_text, top_k=5)

            # Auto-fallback to corrected query if original returns nothing
            if not results and suggested_query:
                print(f"[*] Showing results for suggested query: '{suggested_query}'")
                results = search_core.search(suggested_query, top_k=5)

            if not results:
                print(f"[-] No matching documents found for '{query_text}'.")
            else:
                print(f"\nFound {len(results)} ranked matches:")
                print("-" * 65)
                for rank, item in enumerate(results, start=1):
                    print(f"[{rank}] {item.title} (Score: {item.score})")
                    if item.url:
                        print(f"    URL: {item.url}")
                    print(f"    Snippet: {item.snippet}")
                    print("-" * 65)

        elif action in ("2", "ingest"):
            target_path = parameter.strip() or DEFAULT_DATA_FOLDER
            if not os.path.exists(target_path):
                print(f"[!] Path does not exist: '{target_path}'")
                continue

            if os.path.isfile(target_path):
                if target_path.endswith(".txt"):
                    is_ok = ingestion_service.ingest_text_file(target_path)
                    print(f"[+] Text file ingested: {is_ok}")
                elif target_path.endswith(".json"):
                    count = ingestion_service.ingest_json_file(target_path)
                    print(f"[+] Ingested {count} documents from JSON file.")
                else:
                    print("[!] Unsupported file format. Please provide .txt or .json")
            else:
                total_loaded = ingestion_service.ingest_directory(target_path)
                print(f"[+] Ingested {total_loaded} total documents from directory.")

        elif action in ("3", "save"):
            save_dest = parameter.strip() or DEFAULT_STORAGE_FILE
            index_engine.save_to_disk(save_dest)
            print(f"[+] Inverted index successfully saved to: '{save_dest}'")

        elif action in ("4", "load"):
            load_dest = parameter.strip() or DEFAULT_STORAGE_FILE
            try:
                index_engine.load_from_disk(load_dest)
                print(f"[+] Index loaded successfully ({index_engine.total_documents} documents).")
            except Exception as read_err:
                print(f"[!] Error reading index file: {read_err}")

        elif action in ("5", "stats"):
            vocab_size = len(index_engine.index)
            doc_count = index_engine.total_documents
            print("\n--- Corpus & Index Metrics ---")
            print(f"Total Documents Indexed: {doc_count}")
            print(f"Unique Vocabulary Terms: {vocab_size}")
            print("------------------------------")

        else:
            print(f"[!] Unrecognized command: '{action}'. Please choose from 1 to 6.")


if __name__ == "__main__":
    run_cli()