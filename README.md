# LDRS v2 — Living Document RAG System

A filesystem-native, tree-aware retrieval-augmented generation system for
structured PDF documents. LDRS v2 replaces traditional vector similarity
search with hierarchical document navigation, letting an LLM search through
document structure trees and extract context from precisely the right
sections.

## Key Design Decisions

- **No vector database.** Documents are indexed into structure JSON files
  (from PageIndex) and converted to structured Markdown. Search is done
  via regex/keyword matching against titles, summaries, and body text in
  the document tree.
- **UTF-8 NFC normalization** everywhere — critical for Nepali/Devanagari
  text safety.
- **LLM-in-the-loop** for query expansion, document selection, and answer
  generation. Retrieval (grep + context fetching) is deterministic.
- **Multi-document** — queries can span multiple documents in a corpus.

## Architecture

```
[Query Time]

  User Query
    │
    ├─→ QueryExpander (LLM)      → 3-5 sub-queries
    │
    ├─→ DocSelector (LLM)        → pick relevant docs from corpus
    │
    ├─→ Per-doc (parallel):
    │     ├─→ TreeGrep            → search titles/summaries/body
    │     └─→ ContextFetcher      → extract .md body for matched nodes
    │
    ├─→ ContextMerger             → rank, dedup, budget across all docs
    │
    └─→ Generator (LLM)          → answer with cross-doc citations
         │
         └─→ LDRSResult

[Indexing Time]

  PDF + structure JSON
    ├─→ MdExtractor         → structured .md with node metadata
    ├─→ DocRegistry          → corpus TOC (_registry.json)
    └─→ ChangeLog            → corpus file ledger (_changelog.json)
```

## Project Structure

```
ldrs_v2/
├── .env                          # API_KEY, BASE_URL for LLM provider
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Multi-target Docker build
├── .dockerignore                 # Docker build exclusions
├── ldrs_v2_api.py                # FastAPI server (port 8001)
├── streamlit_app.py              # Streamlit web UI
├── CLAUDE.md                     # AI-assisted development guidance
├── README.md                     # This file
│
├── ldrs/                         # Core library
│   ├── __init__.py               # Package exports
│   ├── md_extractor.py           # PDF → structured Markdown
│   ├── doc_telescope.py          # Structural context builder
│   ├── doc_registry.py           # Corpus inventory (_registry.json)
│   ├── changelog.py              # Corpus file ledger (_changelog.json)
│   ├── query_expander.py         # LLM multi-query expansion
│   ├── doc_selector.py           # LLM-based document selection
│   ├── tree_grep.py              # Hierarchical pattern search
│   ├── context_merger.py         # Cross-doc ranking/dedup/budgeting
│   └── ldrs_pipeline.py          # End-to-end orchestrator
│
├── rag/                          # RAG utilities (shared with v1)
│   ├── context_fetcher.py        # Node-level text extraction
│   ├── generator.py              # LLM answer generation
│   ├── retriever.py              # Legacy retriever (v1)
│   └── pipeline.py               # Legacy pipeline (v1)
│
├── pageindex/                    # PageIndex library (read-only dep)
│
├── scripts/
│   └── run_ldrs_query.py         # CLI entry point
│
└── tests/
    ├── test_ldrs_v2.py           # 140 tests (Steps 1-5)
    ├── results/                  # *_structure.json fixture files
    └── pdfs/                     # Test PDF files
```

## Modules

### Step 1 — `ldrs/md_extractor.py`

Converts PDFs into structured Markdown using PyMuPDF (fitz). Each section
gets an HTML comment marker with its `node_id` and page range:

```markdown
<!-- node_id: 0003 | pages: 5-8 -->
### Section Title

Extracted body text...
```

Orphan pages (not covered by any structure node) are appended in an
"Appendix (Uncovered Pages)" section.

### Step 2 — `ldrs/doc_registry.py` + `ldrs/changelog.py`

- **DocRegistry**: Corpus inventory built from `*_structure.json` files.
  Stores doc name, description, node count, page range, top-level sections.
  Provides `to_llm_summary()` for feeding to the document selector.

- **ChangeLog**: File-level ledger tracking `indexed`, `updated`, and
  `deleted` actions with structural diffs and commit hashes.

### Step 3 — `ldrs/query_expander.py` + `ldrs/doc_selector.py`

- **QueryExpander**: LLM-powered expansion of a user query into 3-5
  sub-queries optimized for keyword/regex search. Includes robust JSON
  parsing with fallback heuristics.

- **DocSelector**: LLM picks which documents are relevant to a query.
  Fast paths for single-doc or empty corpora. Fuzzy matching with NFC
  normalization for the LLM's doc-name output.

### Step 4 — `ldrs/tree_grep.py` + `rag/context_fetcher.py` + `ldrs/context_merger.py`

- **TreeGrep**: Three-source search across title, summary, and `.md` body
  text. Supports regex patterns, scope filtering by node_id, and
  multi-pattern deduplication.

- **ContextFetcher**: Extracts text for specific node_ids from `.md` files
  (with fallback to PDF via PyPDF2). Caches parsed `.md` sections.

- **ContextMerger**: Cross-document ranking, deduplication, and character
  budgeting. Groups output by document for readable context.

### Step 5 — `ldrs/ldrs_pipeline.py` + `ldrs_v2_api.py` + `scripts/run_ldrs_query.py`

- **LDRSPipeline**: End-to-end orchestrator. `build_corpus()` scans for
  indexes, `index_document()` ingests a single PDF, `query()` runs the
  5-stage async pipeline, `batch_query()` processes multiple queries.

- **LDRSConfig**: Centralized configuration dataclass for all directories,
  model names, and tuning parameters.

- **LDRSResult**: Query result dataclass with answer, citations, sub-queries,
  selected docs, grep hit count, merged context, timings, and error field.

## Setup

### Prerequisites

- Python 3.12+
- A `.env` file in the project root with LLM credentials:

```env
API_KEY=your_api_key_here
BASE_URL=https://your-llm-provider/v1
```

### Installation

The project uses the virtual environment from the parent directory:

```bash
# Activate the venv
source /Users/urgensingtan/Desktop/PageIndexlocal/.venv/bin/activate

# Install any missing dependencies (use --target since venv has no pip)
pip install --target "$(python -c 'import site; print(site.getsitepackages()[0])')" \
    pymupdf openai python-dotenv fastapi uvicorn pytest-asyncio
```

### Running Tests

```bash
source /Users/urgensingtan/Desktop/PageIndexlocal/.venv/bin/activate
cd /Users/urgensingtan/Desktop/PageIndexlocal/ldrs_v2
python -m pytest tests/test_ldrs_v2.py -v
```

All 140 tests should pass. Tests are fully offline — no LLM calls. Async
tests use `pytest-asyncio` with `unittest.mock.AsyncMock`.

## Usage

### CLI

```bash
# Single query
python scripts/run_ldrs_query.py --query "What is Earth Mover's Distance?" --verbose

# Interactive mode
python scripts/run_ldrs_query.py --interactive

# Show corpus summary
python scripts/run_ldrs_query.py --corpus

# JSON output
python scripts/run_ldrs_query.py --query "..." --json-output
```

### Python API

```python
import asyncio
from ldrs import LDRSPipeline, LDRSConfig

config = LDRSConfig(
    results_dir="tests/results",
    pdf_dir="tests/pdfs",
    model="qwen3-vl",
)
pipeline = LDRSPipeline(config)
pipeline.build_corpus()

result = asyncio.run(pipeline.query("What is the main topic?"))
print(result.answer)
print(result.citations)
```

### REST API

```bash
# Start the server
python ldrs_v2_api.py

# Query
curl -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is Earth Mover'\''s Distance?"}'

# Corpus info
curl http://localhost:8001/corpus
curl http://localhost:8001/corpus/stats

# Index a new document
curl -X POST http://localhost:8001/index \
  -H 'Content-Type: application/json' \
  -d '{"pdf_path": "/path/to/doc.pdf", "index_path": "/path/to/doc_structure.json"}'

# Health check
curl http://localhost:8001/health
```

### Streamlit Web UI

```bash
source /Users/urgensingtan/Desktop/PageIndexlocal/.venv/bin/activate
cd /Users/urgensingtan/Desktop/PageIndexlocal/ldrs_v2
streamlit run streamlit_app.py
```

Opens at [http://localhost:8501](http://localhost:8501). Features:

- **Query tab** — Enter a question, view the answer with citations,
  sub-queries, selected documents, context stats, timing breakdown, and
  LLM reasoning. Query history is preserved within the session.
- **Corpus tab** — View corpus statistics, browse all registered documents
  with their metadata (node count, page range, top-level sections), and
  rebuild the corpus from disk.
- **Index Document tab** — Index a new PDF by providing its file path
  and structure JSON path. Includes a quick-index selector for test fixtures.
- **Batch Query tab** — Run multiple queries in sequence (one per line)
  and view all results.
- **Sidebar** — Configure directories, LLM model, and all pipeline tuning
  parameters (sub-query counts, grep limits, context budgets, etc.).
  Click "Apply Config & Rebuild Pipeline" to apply changes.

### Docker

The project includes a multi-target Dockerfile with three targets: **api**
(default), **streamlit**, and **cli**.

```bash
# Build the API server image (default target)
docker build -t ldrs-v2 .

# Build the Streamlit UI image
docker build -t ldrs-v2-ui --target streamlit .

# Build the CLI image
docker build -t ldrs-v2-cli --target cli .
```

**Running containers** — pass your `.env` file via `--env-file` (it is
excluded from the image by `.dockerignore`):

```bash
# FastAPI server (port 8001)
docker run --env-file .env -p 8001:8001 ldrs-v2

# Streamlit web UI (port 8501)
docker run --env-file .env -p 8501:8501 ldrs-v2-ui

# CLI single query
docker run --env-file .env ldrs-v2-cli --query "What is Earth Mover's Distance?"

# CLI interactive mode
docker run -it --env-file .env ldrs-v2-cli
```

**Custom data** — mount your own PDFs and structure JSONs:

```bash
docker run --env-file .env \
  -v /path/to/your/results:/app/data/results \
  -v /path/to/your/pdfs:/app/data/pdfs \
  -e LDRS_RESULTS_DIR=/app/data/results \
  -e LDRS_PDF_DIR=/app/data/pdfs \
  -p 8001:8001 ldrs-v2
```

The test data (9 PDFs + structure JSONs) is bundled in the image so it
works out of the box without any mounts.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Run a query through the full pipeline |
| POST | `/batch-query` | Run multiple queries sequentially |
| GET | `/corpus` | Get corpus summary (registry + changelog) |
| GET | `/corpus/stats` | Get corpus statistics |
| POST | `/corpus/rebuild` | Rebuild the corpus registry from disk |
| POST | `/index` | Index a single document |
| GET | `/health` | Health check |

## Configuration

`LDRSConfig` accepts these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `results_dir` | (required) | Directory with `*_structure.json` files |
| `pdf_dir` | (required) | Directory with source PDFs |
| `md_dir` | `results_dir` | Directory for cached `.md` files |
| `registry_path` | auto | Path to `_registry.json` |
| `changelog_path` | auto | Path to `_changelog.json` |
| `model` | `"qwen3-vl"` | LLM model name |
| `max_sub_queries` | `5` | Max sub-queries from QueryExpander |
| `max_grep_results` | `20` | Max grep hits per document |
| `max_total_chars` | `12000` | Character budget for merged context |
| `max_chunks` | `15` | Max chunks in merged context |
| `max_chars_per_node` | `3000` | Max chars per node from ContextFetcher |

## PageIndex JSON Schema

LDRS v2 consumes structure JSON files produced by PageIndex:

```json
{
  "doc_name": "earthmover.pdf",
  "doc_description": "Optional document description",
  "structure": [
    {
      "title": "Section Title",
      "start_index": 1,
      "end_index": 5,
      "node_id": "0001",
      "summary": "Optional section summary",
      "nodes": [
        {
          "title": "Subsection",
          "start_index": 2,
          "end_index": 3,
          "node_id": "0002"
        }
      ]
    }
  ]
}
```

- `start_index` / `end_index` are 1-based page numbers.
- `node_id` is a zero-padded 4-digit string.
- `doc_description` is optional (only present in some documents).
- `summary` is optional per node.
- `nodes` contains recursive children.
