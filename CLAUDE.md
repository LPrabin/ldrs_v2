# CLAUDE.md — LDRS v2 Development Guidance

This file provides context for AI-assisted development on the LDRS v2
project. It is designed to be read by language models (Claude, GPT, etc.)
at the start of a coding session.

## Project Identity

- **Name**: LDRS v2 (Living Document RAG System, version 2)
- **Location**: `/Users/urgensingtan/Desktop/PageIndexlocal/ldrs_v2/`
- **Language**: Python 3.12
- **Test framework**: pytest + pytest-asyncio

## Environment

- The virtual environment lives in the **parent directory**:
  `/Users/urgensingtan/Desktop/PageIndexlocal/.venv/` (symlink to
  `/Users/urgensingtan/Desktop/PageIndex/.venv/`)
- Activate: `source /Users/urgensingtan/Desktop/PageIndexlocal/.venv/bin/activate`
- The venv does **not** have its own pip. Install packages with:
  ```bash
  pip install --target "$(python -c 'import site; print(site.getsitepackages()[0])')" <package>
  ```
- All scripts need `sys.path.insert(0, PROJECT_ROOT)` to find project
  modules (where `PROJECT_ROOT` is the ldrs_v2 directory).

## Running Tests

```bash
source /Users/urgensingtan/Desktop/PageIndexlocal/.venv/bin/activate
cd /Users/urgensingtan/Desktop/PageIndexlocal/ldrs_v2
python -m pytest tests/test_ldrs_v2.py -v
```

- **140 tests** covering Steps 1-5 (all passing).
- Tests are fully offline — no LLM calls.
- Async tests use `pytest-asyncio` (v1.3.0) with `unittest.mock.AsyncMock`.
- Test fixtures use `tmp_path` / `tmp_output_dir` for isolation.
- Test data: `tests/results/` (9 structure JSONs), `tests/pdfs/` (9 PDFs).

## LLM Integration Pattern

All LLM calls use `openai.AsyncOpenAI`:

```python
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.AsyncOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
```

- `.env` file in project root provides `API_KEY` and `BASE_URL`.
- Default model: `"qwen3-vl"` (configurable via `LDRSConfig.model`).
- QueryExpander and DocSelector use system + user messages.
- Generator uses a single user message, `temperature=0`.

## Critical Conventions

### UTF-8 NFC Normalization

Every text boundary must apply `unicodedata.normalize("NFC", text)` before
matching or storage. This is non-negotiable for Nepali/Devanagari text:

```python
import unicodedata
text = unicodedata.normalize("NFC", text)
```

### Markdown Format

Extracted `.md` files use HTML comments as section markers:

```markdown
<!-- doc_name: earthmover.pdf -->
<!-- doc_description: Optional description -->

<!-- node_id: 0001 | pages: 1-5 -->
# Section Title

Body text...

<!-- node_id: 0002 | pages: 2-3 -->
## Subsection

More text...
```

The parser regex: `re.compile(r"<!--\s*node_id:\s*(\S+)\s*\|")`

### PageIndex JSON Schema

```json
{
  "doc_name": "filename.pdf",
  "doc_description": "optional",
  "structure": [{
    "title": "Section",
    "start_index": 1,
    "end_index": 5,
    "node_id": "0001",
    "summary": "optional",
    "nodes": [...]
  }]
}
```

- `start_index`/`end_index` are 1-based page numbers.
- `node_id` is zero-padded 4-digit string (e.g., `"0001"`).
- `doc_description` is optional (only ~3 of 10 test docs have it).
- `summary` is optional per node.

### ContextFetcher Output Header Format

```
## Section: Title (Pages X-Y) [node_id: XXXX]
```

This format is parsed by `ContextMerger._parse_fetcher_header()` and
`LDRSPipeline._add_to_merger()`.

### Logging

All modules use `logging.DEBUG` level extensively. The standard pattern:

```python
import logging
logger = logging.getLogger(__name__)
logger.debug("ClassName.method  key=%s  value=%s", key, value)
```

## Module Dependency Graph

```
ldrs_pipeline
  ├── query_expander    (LLM)
  ├── doc_selector      (LLM)
  ├── doc_registry
  ├── changelog
  ├── tree_grep
  ├── context_merger
  ├── md_extractor
  └── rag/
      ├── context_fetcher
      └── generator     (LLM)
```

Only three modules make LLM calls: `query_expander`, `doc_selector`,
and `rag/generator`. All others are deterministic.

## Key Implementation Details

### Lazy Initialization

LLM-calling components in the pipeline are created on first use:

```python
@property
def query_expander(self) -> QueryExpander:
    if self._query_expander is None:
        self._query_expander = QueryExpander(model=self.config.model)
    return self._query_expander
```

This allows setting mock objects for testing:
```python
pipeline._query_expander = AsyncMock()
```

### DocSelector Fast Paths

- Empty corpus → empty selection (no LLM call)
- Single document → auto-select it (no LLM call)
- Fuzzy matching on LLM output (case-insensitive, substring)

### ContextMerger Budget

- `max_total_chars` (default 12000) — character budget for all chunks
- `max_chunks` (default 15) — hard cap on number of chunks
- Chunks are ranked by relevance score, then truncated to budget
- Minimum useful chunk size: 100 chars (smaller chunks are dropped)

### TreeGrep Scoring

- Title match: relevance_score = 3.0
- Summary match: relevance_score = 2.0
- Body match: relevance_score = 1.0
- Results sorted by score descending

### ChangeLog vs Query History

The ChangeLog is a **corpus file ledger** — it tracks documents
added/updated/deleted from the corpus. It is NOT a query history log.

## Common Pitfalls

1. **PyMuPDF (fitz)** is the PDF library — not PyPDF2 (which is only
   used as a fallback in `ContextFetcher`). Some PDFs produce fragmented
   text — this is a PDF encoding issue, not a bug.

2. **`add_or_update()`** on DocRegistry takes `(index_path, md_path)`,
   not a pre-built entry dict. It calls `build_entry()` internally.

3. **`record_indexed()`** on ChangeLog takes `(doc_name, index_data, structure)`,
   not `(doc_name, index_path, node_count)`. It computes commit hashes
   and node counts internally.

4. **MdExtractor** has two extraction methods:
   - `extract_to_string()` → returns markdown as string
   - `extract(output_filename=None)` → writes to file, returns path
   - There is NO `extract_to_file()` method.

5. **LSP errors from parent directory** (`/PageIndexlocal/pageindex/`,
   `/PageIndexlocal/rag/`) are pre-existing and irrelevant to ldrs_v2.

6. **Test search patterns** must match actual structure file titles.
   E.g., the earthmover doc uses "Earth Mover" (two words), not
   "Earthmoving" or "Earthmover".

## Adding a New Module

1. Create the module in `ldrs/`.
2. Add exports to `ldrs/__init__.py` and `__all__`.
3. Add docstrings (module-level + all public classes/methods).
4. Apply NFC normalization at text boundaries.
5. Add `logging.getLogger(__name__)` with DEBUG-level logging.
6. Write tests in `tests/test_ldrs_v2.py`.
7. Run the full test suite to verify no regressions.

## API Endpoints (FastAPI, port 8001)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Full pipeline query |
| POST | `/batch-query` | Multiple queries |
| GET | `/corpus` | Corpus summary |
| GET | `/corpus/stats` | Corpus statistics |
| POST | `/corpus/rebuild` | Rebuild registry |
| POST | `/index` | Index a document |
| GET | `/health` | Health check |
