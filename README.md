# 🔍 Python-Based Search Engine

A lightweight, scalable, and fully custom full-text search engine built in Python. This project implements core Information Retrieval (IR) concepts from scratch, including tokenization, stopword filtering, custom Porter-style stemming, inverted index construction with positional tracking, disk persistence, and **TF-IDF** (Term Frequency-Inverse Document Frequency) relevance scoring.

---

## 🚀 Key Features

- **Custom NLP Pipeline:** Regex-based tokenizer, stopword removal, and zero-dependency morphological stemming.
- **Positional Inverted Index:** Maps terms to document postings with frequency and exact positional occurrences.
- **TF-IDF Relevance Ranking:** Computes document relevance dynamically using smooth IDF and normalized term frequencies.
- **Contextual Snippets:** Automatically generates search snippets highlighting matched query terms in context.
- **Document Ingestion:** Ingests raw `.txt` files or structured `.json` collections recursively from directories.
- **Disk Persistence:** Serializes and deserializes the full index state to and from disk in JSON format.
- **Interactive CLI:** Terminal-based user interface for real-time querying, ingestion, and index inspection.
- **Automated Test Suite:** Comprehensive unit and integration testing via `pytest`.

---

## 🏛️ System Architecture


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
│                    Inverted Index Store                     │
│        Term ──► { Doc_ID: { Frequency, [Positions] } }      │
│                     (JSON Persistence)                      │
└──────────────────────────────┬──────────────────────────────┘
│
┌───────────────┴───────────────┐
▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│        Query Engine         │ │    TF-IDF Ranking Engine    │
│  (Tokenize & Fetch Postings)│ │  Score = Σ (TF × Smooth IDF)│
└──────────────┬──────────────┘ └──────────────┬──────────────┘
│                               │
└───────────────┬───────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│                    Ranked Search Results                    │
│             (Top-K Results + Context Snippets)              │
└─────────────────────────────────────────────────────────────┘


---

## 📐 Scoring Formula

The relevance score of a document $d$ for a query $Q$ is computed as:

$$\text{Score}(Q, d) = \sum_{t \in Q} \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where:
- **Normalized Term Frequency ($\text{TF}$):**
  $$\text{TF}(t, d) = \frac{f_{t,d}}{|d|}$$
  *(where $f_{t,d}$ is the frequency of term $t$ in document $d$, and $|d|$ is the total token count of $d$)*

- **Smooth Inverse Document Frequency ($\text{IDF}$):**
  $$\text{IDF}(t, D) = \ln\left(\frac{|D|}{|\{d \in D : t \in d\}|} + 1.0\right)$$
  *(where $|D|$ is the total number of documents in the index)*

---

## 📂 Project Structure

python-search-engine/
│
├── data/
│   └── documents/              # Local documents storage (git-ignored)
│       └── sample_data.json
├── src/
│   ├── init.py
│   ├── text_processor.py       # Tokenization, stopwords, stemming
│   ├── inverted_index.py       # Postings list & persistence
│   ├── search_engine.py        # TF-IDF ranking & snippet extraction
│   ├── document_ingestion.py   # Text & JSON ingestion handlers
│   └── cli.py                  # Interactive CLI interface
│
├── storage/                    # Serialized index storage (git-ignored)
├── tests/
│   ├── init.py
│   ├── test_text_processor.py
│   ├── test_inverted_index.py
│   ├── test_search_engine.py
│   └── test_document_ingestion.py
│
├── .env.example                # Configuration blueprint
├── .gitignore                  # Git privacy & hygiene rules
├── main.py                     # Application entry point
├── requirements.txt            # Dependency definitions
└── README.md                   # Project documentation


---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- PowerShell, Bash, or Zsh

### 1. Clone the Repository & Setup Environment
```bash
git clone [https://github.com/your-username/python-search-engine.git](https://github.com/your-username/python-search-engine.git)
cd python-search-engine

# Initialize and activate virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows (PowerShell)
# source venv/bin/activate   # Linux / macOS
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Setup Environment Variables
Bash
cp .env.example .env
💻 Usage
Run Interactive CLI
Bash
python main.py
CLI Command Options
Commands:
  1. search <query>       - Search ranked documents
  2. ingest <path>        - Ingest file or folder (default: data/documents)
  3. save [filepath]      - Save current index to disk
  4. load [filepath]      - Load index from disk
  5. stats                - View index metrics & total term count
  6. exit                 - Exit application
🧪 Running Tests
Execute the automated test suite with verbose output:

Bash
pytest -v
🔒 Security & Privacy
Sensitive files, runtime logs, cache binaries, and index storage directories are strictly excluded from version control via .gitignore.

No proprietary API keys or confidential credentials are embedded in code or configuration samples.

📄 License
This project is open-source and available under the MIT License.
