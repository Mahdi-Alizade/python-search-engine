# 🔍 Python-Based Full-Text Search Engine

A high-performance, modular full-text search engine built entirely from scratch in Python. It includes a custom positional inverted index, **Okapi BM25** relevance ranking, exact phrase matching (`"..."`), boolean logic operators (`AND`, `OR`, `NOT`), Levenshtein-based spelling correction (*Did you mean...*), interactive terminal CLI, and an asynchronous **FastAPI REST API**.

---

## 🚀 Key Architectural Features

- **Text Processing Pipeline:** Clean regex-based tokenizer, stopword filtering, and zero-dependency morphological stemming.
- **Positional Inverted Index:** Tracks document frequencies and exact token positions for proximity and phrase verification.
- **Okapi BM25 Ranking:** Modern probabilistic information retrieval model with length normalization ($b=0.75$) and term frequency saturation ($k_1=1.5$).
- **Exact Phrase Search:** Enforces strictly sequential token matching via word position offsets.
- **Boolean Query Engine:** Supports logical set operations (`AND`, `OR`, `NOT`) directly over index postings.
- **Spell Correction & Typo Tolerance:** Suggests vocabulary corrections using dynamic programming Levenshtein distance.
- **Document Ingestion:** Ingests raw `.txt` files and structured `.json` collections recursively.
- **Disk Persistence:** Serializes and restores full corpus indices to/from disk in portable JSON format.
- **Dual Interfaces:** Full interactive terminal CLI and production-ready FastAPI REST API with OpenAPI/Swagger docs.
- **Automated Test Suite:** Comprehensive coverage across all engine layers with 45+ unit and integration tests.

---

## 🏛️ System Pipeline

┌─────────────────────────────────────────────────────────────┐
│                       Text Corpus                           │
│                 (.txt files / .json datasets)               │
└──────────────────────────────┬──────────────────────────────┘
│ Ingestion
▼
┌─────────────────────────────────────────────────────────────┐
│                    Text Processor Pipeline                  │
│    Tokenization  ──►  Stopword Filtering  ──►  Stemming     │
└──────────────────────────────┬──────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│                 Positional Inverted Index                   │
│        Term ──► { Doc_ID: { Frequency, [Positions] } }      │
│                     (JSON Persistence)                      │
└──────────────┬──────────────────────────────┬───────────────┘
│                              │
▼                              ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│     Phrase & Boolean Engine │ │       Okapi BM25 Scorer     │
│  (Positional & Set Filters) │ │  (Length Normalized Metric) │
└──────────────┬──────────────┘ └──────────────┬──────────────┘
│                               │
└───────────────┬───────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│                 Spell Corrector & Snippets                  │
│         (Levenshtein Distance + Context Extraction)         │
└──────────────────────────────┬──────────────────────────────┘
│
┌───────────────┴───────────────┐
▼                               ▼
Terminal CLI App                 FastAPI REST API


---

## 📂 Project Structure

python-search-engine/
│
├── data/
│   └── documents/              # Local corpus files (git-ignored)
│       └── sample_data.json
├── src/
│   ├── init.py
│   ├── models.py               # DocumentRecord & TermPosting data models
│   ├── tokenizer.py            # Tokenizer, stopword removal & stemmer
│   ├── text_processor.py       # General text processor pipeline
│   ├── inverted_index.py       # Positional inverted index & serialization
│   ├── bm25_scorer.py          # Okapi BM25 scoring formulation
│   ├── phrase_matcher.py       # Positional exact phrase matching
│   ├── boolean_parser.py       # Boolean logic (AND, OR, NOT) evaluator
│   ├── spell_corrector.py      # Levenshtein distance typo corrector
│   ├── search_engine.py        # Unified query engine & snippet builder
│   ├── document_ingestion.py   # Text & JSON batch ingestion handler
│   ├── cli.py                  # Terminal CLI interactive interface
│   ├── api_schemas.py          # Pydantic schemas for API endpoints
│   └── api.py                  # FastAPI REST API web service
│
├── storage/                    # Serialized index storage (git-ignored)
├── tests/
│   ├── test_models.py
│   ├── test_tokenizer.py
│   ├── test_inverted_index.py
│   ├── test_bm25_scorer.py
│   ├── test_phrase_matcher.py
│   ├── test_boolean_parser.py
│   ├── test_spell_corrector.py
│   ├── test_search_engine.py
│   ├── test_document_ingestion.py
│   └── test_api.py
│
├── .env.example
├── .gitignore
├── main.py
├── requirements.txt
└── README.md


---

## 🛠️ Installation & Setup

### 1. Initialize Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
2. Install Dependencies
PowerShell
pip install -r requirements.txt
3. Setup Environment Variables
PowerShell
cp .env.example .env
💻 Running the Search Engine
Option A: Interactive Terminal CLI
PowerShell
python main.py
Supported CLI queries:

Standard Search: search distributed systems

Exact Phrase: search "asynchronous programming"

Boolean AND: search Python AND asyncio

Boolean NOT: search Python NOT backend

Auto-Correction: Searching for pythn will trigger 💡 Did you mean: python

Option B: FastAPI REST Service
Launch the async API server:

PowerShell
uvicorn src.api:app --reload --port 8000
Interactive Swagger Documentation: http://127.0.0.1:8000/docs

System Health Check:

Bash
curl [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
Execute Search Query:

Bash
curl "[http://127.0.0.1:8000/api/v1/search?q=FastAPI&limit=5](http://127.0.0.1:8000/api/v1/search?q=FastAPI&limit=5)"
Index New Document on the Fly:

Bash
curl -X POST "[http://127.0.0.1:8000/api/v1/documents](http://127.0.0.1:8000/api/v1/documents)" \
     -H "Content-Type: application/json" \
     -d '{"doc_id": "doc_cloud", "title": "Cloud Computing", "content": "Elastic compute clusters in the cloud."}'
🧪 Running Automated Tests
Run the full suite of 45+ unit and integration tests:

PowerShell
pytest -v
🔒 Security & Privacy
All local data dumps, persistent index JSON binaries, virtual environment directories, and .env files are strictly excluded from version control via .gitignore.

No sensitive keys or external cloud dependencies are required.

📄 License
This project is open-source and licensed under the MIT License.