"""
LDRS v2 API — FastAPI server for the Living Document RAG System.

Provides HTTP endpoints for querying the LDRS v2 pipeline and managing
the document corpus.

Endpoints:
    POST /query              — Run a query through the full pipeline.
    POST /batch-query        — Run multiple queries sequentially.
    GET  /corpus             — Get corpus summary.
    GET  /corpus/stats       — Get corpus statistics.
    POST /corpus/rebuild     — Rebuild the corpus registry.
    POST /index              — Index a single document (extract .md + register).
    POST /index-pdf          — Index a raw PDF (PageIndex + extract .md + register).
    GET  /providers           — List available LLM providers.
    GET  /health             — Health check.

Usage::

    # Start the server (port 8001)
    python ldrs_v2_api.py

    # Or with uvicorn directly
    uvicorn ldrs_v2_api:app --host 0.0.0.0 --port 8001 --reload

    # Query the API
    curl -X POST http://localhost:8001/query \\
        -H 'Content-Type: application/json' \\
        -d '{"query": "What is Earth Mover\\'s Distance?"}'

    # Query with a specific provider
    curl -X POST http://localhost:8001/query \\
        -H 'Content-Type: application/json' \\
        -d '{"query": "What is EMD?", "provider": "openai"}'
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ldrs.ldrs_pipeline import LDRSConfig, LDRSPipeline, LDRSResult
from ldrs.llm_provider import (
    LLMProvider,
    get_provider,
    clear_provider_cache,
    list_available_providers,
    get_provider_info,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Quiet noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config from environment (override via env vars)
# ---------------------------------------------------------------------------
RESULTS_DIR = os.getenv("LDRS_RESULTS_DIR", "tests/results")
PDF_DIR = os.getenv("LDRS_PDF_DIR", "tests/pdfs")
MD_DIR = os.getenv("LDRS_MD_DIR", None)
MODEL = os.getenv("LDRS_MODEL", "qwen3-vl")
PORT = int(os.getenv("LDRS_PORT", "8001"))

# ---------------------------------------------------------------------------
# Pydantic models (request/response schemas)
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for /query."""

    query: str = Field(..., description="The natural-language question.")
    model: Optional[str] = Field(None, description="Override LLM model (optional).")
    provider: Optional[str] = Field(
        None,
        description=(
            "LLM provider to use for this query ('local', 'openai', 'gemini'). "
            "If omitted, uses the server's default provider."
        ),
    )


class BatchQueryRequest(BaseModel):
    """Request body for /batch-query."""

    queries: List[str] = Field(..., description="List of questions.")


class QueryResponse(BaseModel):
    """Response body for /query."""

    query: str
    answer: str
    sub_queries: List[str]
    selected_docs: List[str]
    grep_hits: int
    citations: List[Dict[str, Any]]
    timings: Dict[str, float]
    error: str = ""
    merged_context_stats: Optional[Dict[str, Any]] = None
    usage_stats: Optional[Dict[str, Any]] = None


class IndexRequest(BaseModel):
    """Request body for /index."""

    pdf_path: str = Field(..., description="Path to the PDF file.")
    index_path: str = Field(..., description="Path to the *_structure.json file.")
    md_filename: Optional[str] = Field(
        None, description="Custom .md filename (optional)."
    )
    use_ocr: bool = Field(
        False, description="Extract text using OCR rather than font-based extraction."
    )


class IndexPdfRequest(BaseModel):
    """Request body for /index-pdf (raw PDF → PageIndex → register)."""

    pdf_path: str = Field(..., description="Path to the PDF file.")
    output_dir: Optional[str] = Field(
        None,
        description=(
            "Directory to save structure JSON and .md files. "
            "Defaults to the pipeline's results_dir."
        ),
    )
    md_filename: Optional[str] = Field(
        None, description="Custom .md filename (optional)."
    )
    use_ocr: bool = Field(
        False,
        description="Extract text using OCR rather than font-based extraction. Essential for documents containing Devanagari script.",
    )


class CorpusStats(BaseModel):
    """Response body for /corpus/stats."""

    num_documents: int
    doc_names: List[str]
    changelog_entries: int
    active_docs: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_response(result: LDRSResult) -> QueryResponse:
    """Convert an LDRSResult to a QueryResponse."""
    merged_stats = None
    if result.merged_context:
        mc = result.merged_context
        merged_stats = {
            "num_chunks": mc.num_chunks,
            "total_chars": mc.total_chars,
            "num_docs": mc.num_docs,
            "dropped_count": mc.dropped_count,
        }

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        sub_queries=result.sub_queries,
        selected_docs=result.selected_docs,
        grep_hits=result.grep_hits,
        citations=result.citations,
        timings=result.timings,
        error=result.error,
        merged_context_stats=merged_stats,
        usage_stats=getattr(result, "usage_stats", None),
    )


# ---------------------------------------------------------------------------
# App + pipeline initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LDRS v2 API",
    description=(
        "Living Document RAG System v2 — "
        "Filesystem-native, tree-aware document retrieval and generation."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (initialised on startup)
pipeline: Optional[LDRSPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Initialise the pipeline and build the corpus on startup."""
    global pipeline

    logger.info("LDRS v2 API starting up...")
    logger.info(
        "Config: results_dir=%s  pdf_dir=%s  model=%s  port=%d",
        RESULTS_DIR,
        PDF_DIR,
        MODEL,
        PORT,
    )

    # Log available LLM providers
    available = list_available_providers()
    logger.info("Available LLM providers: %s", ", ".join(available) or "(none)")
    all_info = get_provider_info()
    for pname in available:
        info = all_info.get(pname, {})
        logger.info(
            "  %s: model=%s  has_base_url=%s",
            pname,
            info.get("model", "N/A"),
            info.get("has_base_url", False),
        )

    config = LDRSConfig(
        results_dir=RESULTS_DIR,
        pdf_dir=PDF_DIR,
        md_dir=MD_DIR,
        model=MODEL,
    )

    pipeline = LDRSPipeline(config)

    t0 = time.monotonic()
    count = pipeline.build_corpus()
    elapsed = time.monotonic() - t0

    logger.info("Corpus built: %d documents in %.2fs", count, elapsed)
    logger.info(
        "Default provider: %s (model: %s)",
        pipeline.llm_provider.config.name,
        pipeline.llm_provider.model,
    )
    logger.info("LDRS v2 API ready on port %d", PORT)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "ldrs-v2",
        "corpus_docs": len(pipeline.registry.doc_names) if pipeline else 0,
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Run a query through the full LDRS v2 pipeline.

    The pipeline flow:
    1. Query expansion (3-5 sub-queries)
    2. Document selection (pick relevant docs)
    3. Per-doc retrieval (TreeGrep + ContextFetcher)
    4. Context merging (rank, dedup, budget)
    5. Answer generation (LLM)

    Optionally pass ``provider`` ("local", "openai", "gemini") to use
    a specific LLM provider for this query.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    logger.info(
        "POST /query  query=%r  provider=%s",
        request.query,
        request.provider or "(default)",
    )

    # Temporarily switch provider if requested
    original_provider = None
    if request.provider:
        try:
            new_provider = get_provider(
                provider_name=request.provider,
                model_override=request.model,
            )
            original_provider = pipeline.llm_provider
            pipeline.llm_provider = new_provider
            # Update lazy-init components so they use the new provider
            pipeline._query_expander = None
            pipeline._doc_selector = None
            pipeline._generator = None
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider '{request.provider}': {e}",
            )

    try:
        result = await pipeline.query(request.query)
        return _result_to_response(result)
    finally:
        # Restore original provider if we switched
        if original_provider is not None:
            pipeline.llm_provider = original_provider
            pipeline._query_expander = None
            pipeline._doc_selector = None
            pipeline._generator = None


@app.post("/batch-query", response_model=List[QueryResponse])
async def batch_query_endpoint(request: BatchQueryRequest):
    """Run multiple queries sequentially."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    logger.info("POST /batch-query  queries=%d", len(request.queries))
    results = await pipeline.batch_query(request.queries)
    return [_result_to_response(r) for r in results]


@app.get("/corpus")
async def get_corpus():
    """Get the LLM-readable corpus summary."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    return {"summary": pipeline.corpus_summary()}


@app.get("/corpus/stats", response_model=CorpusStats)
async def get_corpus_stats():
    """Get corpus statistics."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    stats = pipeline.corpus_stats()
    return CorpusStats(**stats)


@app.post("/corpus/rebuild")
async def rebuild_corpus():
    """Rebuild the corpus registry by re-scanning the results directory."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    t0 = time.monotonic()
    count = pipeline.build_corpus()
    elapsed = time.monotonic() - t0

    return {
        "status": "ok",
        "documents_registered": count,
        "elapsed_seconds": round(elapsed, 3),
    }


@app.post("/index")
async def index_document(request: IndexRequest):
    """
    Index a single document: extract .md, register in the corpus, and
    log to the changelog.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    if not os.path.exists(request.pdf_path):
        raise HTTPException(
            status_code=404, detail=f"PDF not found: {request.pdf_path}"
        )
    if not os.path.exists(request.index_path):
        raise HTTPException(
            status_code=404,
            detail=f"Index not found: {request.index_path}",
        )

    try:
        md_path = pipeline.index_document(
            pdf_path=request.pdf_path,
            index_path=request.index_path,
            md_filename=request.md_filename,
            use_ocr=request.use_ocr,
        )
        return {
            "status": "ok",
            "md_path": md_path,
            "corpus_docs": len(pipeline.registry.doc_names),
        }
    except Exception as e:
        logger.exception("POST /index  error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-pdf")
async def index_pdf_endpoint(request: IndexPdfRequest):
    """
    Index a raw PDF: extract markdown via PdfExtractor (font-based heading
    detection), parse headings into a structure tree via md_to_tree, map
    line numbers to page numbers, register in the corpus, and log to the
    changelog.

    This is the full pipeline from raw PDF to corpus-ready document.
    No heavy LLM calls are needed unless summaries are requested.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    if not os.path.exists(request.pdf_path):
        raise HTTPException(
            status_code=404, detail=f"PDF not found: {request.pdf_path}"
        )

    output_dir = request.output_dir or pipeline.config.results_dir
    logger.info("POST /index-pdf  pdf=%s  output_dir=%s", request.pdf_path, output_dir)

    try:
        md_path = await pipeline.index_document_from_pdf(
            pdf_path=request.pdf_path,
            output_dir=output_dir,
            md_filename=request.md_filename,
            use_ocr=request.use_ocr,
        )
        return {
            "status": "ok",
            "md_path": md_path,
            "corpus_docs": len(pipeline.registry.doc_names),
        }
    except Exception as e:
        logger.exception("POST /index-pdf  error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/providers")
async def providers_endpoint():
    """
    List available LLM providers and their configuration status.

    Returns all supported providers with info about which ones are
    properly configured and ready to use.
    """
    available = list_available_providers()
    all_info = get_provider_info()

    # Add the current default provider
    default_provider = None
    if pipeline:
        default_provider = pipeline.llm_provider.provider_name

    return {
        "default_provider": default_provider,
        "available": available,
        "providers": all_info,
    }


# ---------------------------------------------------------------------------
# Run with uvicorn
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ldrs_v2_api:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )
