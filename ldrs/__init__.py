"""
LDRS v2 — Living Document RAG System

Modules (built incrementally):
  - llm_provider:   Centralized multi-provider LLM client (LiteLLM)
  - pdf_extractor:  PDF → Markdown with font-based heading detection (Step 1)
  - doc_telescope:  Structural context builder (carried from v1)
  - doc_registry:   Corpus inventory / TOC                        (Step 2)
  - changelog:      Corpus file ledger                             (Step 2)
  - query_expander: LLM multi-query expansion                     (Step 3)
  - doc_selector:   LLM-based document selection                  (Step 3)
  - tree_grep:      Hierarchical pattern search                   (Step 4)
  - context_merger: Cross-doc context ranking/dedup               (Step 4)
  - ldrs_pipeline:  End-to-end v2 orchestration                   (Step 5)
"""

from ldrs.llm_provider import (
    LLMProvider,
    ProviderConfig,
    get_provider,
    clear_provider_cache,
    list_available_providers,
    get_provider_info,
)
from ldrs.pdf_extractor import (
    PdfExtractor,
    extract_pdf_to_markdown,
    build_line_to_page_map,
    map_structure_pages,
)
from ldrs.doc_telescope import DocTelescope, TelescopeView
from ldrs.doc_registry import DocRegistry, build_entry
from ldrs.changelog import ChangeLog, compute_structural_diff
from ldrs.query_expander import QueryExpander, ExpandedQuery, expand_query
from ldrs.doc_selector import DocSelector, DocSelection, select_documents
from ldrs.tree_grep import TreeGrep, GrepResult
from ldrs.context_merger import ContextMerger, ContextChunk, MergedContext
from ldrs.ldrs_pipeline import LDRSPipeline, LDRSConfig, LDRSResult

__all__ = [
    # LLM Provider
    "LLMProvider",
    "ProviderConfig",
    "get_provider",
    "clear_provider_cache",
    "list_available_providers",
    "get_provider_info",
    # Step 1 — PDF extraction
    "PdfExtractor",
    "extract_pdf_to_markdown",
    "build_line_to_page_map",
    "map_structure_pages",
    # v1 carry-over
    "DocTelescope",
    "TelescopeView",
    # Step 2
    "DocRegistry",
    "build_entry",
    "ChangeLog",
    "compute_structural_diff",
    # Step 3
    "QueryExpander",
    "ExpandedQuery",
    "expand_query",
    "DocSelector",
    "DocSelection",
    "select_documents",
    # Step 4
    "TreeGrep",
    "GrepResult",
    "ContextMerger",
    "ContextChunk",
    "MergedContext",
    # Step 5
    "LDRSPipeline",
    "LDRSConfig",
    "LDRSResult",
]
