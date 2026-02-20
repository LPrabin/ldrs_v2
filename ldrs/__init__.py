"""
LDRS v2 — Living Document RAG System

Modules (built incrementally):
  - md_extractor:   PDF → structured Markdown with node metadata  (Step 1)
  - doc_telescope:  Structural context builder (carried from v1)
  - doc_registry:   Corpus inventory / TOC                        (Step 2)
  - changelog:      Corpus file ledger                             (Step 2)
  - query_expander: LLM multi-query expansion                     (Step 3)
  - doc_selector:   LLM-based document selection                  (Step 3)
  - tree_grep:      Hierarchical pattern search                   (Step 4)
  - context_merger: Cross-doc context ranking/dedup               (Step 4)
  - ldrs_pipeline:  End-to-end v2 orchestration                   (Step 5)
"""

from ldrs.md_extractor import MdExtractor, extract_markdown
from ldrs.doc_telescope import DocTelescope, TelescopeView
from ldrs.doc_registry import DocRegistry, build_entry
from ldrs.changelog import ChangeLog, compute_structural_diff
from ldrs.query_expander import QueryExpander, ExpandedQuery, expand_query
from ldrs.doc_selector import DocSelector, DocSelection, select_documents
from ldrs.tree_grep import TreeGrep, GrepResult
from ldrs.context_merger import ContextMerger, ContextChunk, MergedContext
from ldrs.ldrs_pipeline import LDRSPipeline, LDRSConfig, LDRSResult

__all__ = [
    # Step 1
    "MdExtractor",
    "extract_markdown",
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
