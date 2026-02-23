#!/usr/bin/env python3
"""
LDRS v2 — Streamlit UI

A web interface for the Living Document RAG System (LDRS v2).
Provides interactive querying, corpus management, document indexing,
and full pipeline transparency.

Usage:
    source /Users/urgensingtan/Desktop/PageIndexlocal/.venv/bin/activate
    cd /Users/urgensingtan/Desktop/PageIndexlocal/ldrs_v2
    streamlit run streamlit_app.py
"""

import asyncio
import json
import logging
import os
import sys
import time

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ldrs.ldrs_pipeline import LDRSConfig, LDRSPipeline, LDRSResult
from ldrs.llm_provider import list_available_providers, get_provider_info

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet noisy loggers
for _name in ("httpx", "httpcore", "openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Async helper — run coroutines in Streamlit's synchronous context
# ---------------------------------------------------------------------------


def run_async(coro):
    """Run an async coroutine from synchronous Streamlit code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside a running loop (e.g. Jupyter, some Streamlit setups).
        # Create a new loop in a thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LDRS v2 — Living Document RAG",
    page_icon=":",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_session_state():
    """Initialise session state variables."""
    defaults = {
        "pipeline": None,
        "corpus_built": False,
        "query_history": [],  # List of (query, LDRSResult) tuples
        "results_dir": "tests/results",
        "pdf_dir": "tests/pdfs",
        "md_dir": "",
        "model": "qwen3-vl",
        "provider": None,  # None → use LLM_PROVIDER env var default
        "max_sub_queries": 5,
        "min_sub_queries": 3,
        "max_grep_results": 30,
        "max_total_chars": 15_000,
        "max_chunks": 30,
        "max_chars_per_node": 4000,
        "max_context_tokens": 4000,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ---------------------------------------------------------------------------
# Sidebar — Configuration
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render the configuration sidebar."""
    st.sidebar.title("LDRS v2 Config")

    st.sidebar.subheader("Directories")
    st.session_state.results_dir = st.sidebar.text_input(
        "Results directory",
        value=st.session_state.results_dir,
        help="Directory containing *_structure.json index files.",
    )
    st.session_state.pdf_dir = st.sidebar.text_input(
        "PDF directory",
        value=st.session_state.pdf_dir,
        help="Directory containing source PDF files.",
    )
    st.session_state.md_dir = st.sidebar.text_input(
        "Markdown directory",
        value=st.session_state.md_dir,
        help="Directory for cached .md files (blank = same as results dir).",
    )

    st.sidebar.subheader("LLM Settings")

    # Provider selector
    available_providers = list_available_providers()
    provider_options = ["(default)"] + available_providers
    current_idx = 0
    if st.session_state.provider and st.session_state.provider in available_providers:
        current_idx = provider_options.index(st.session_state.provider)
    selected_provider = st.sidebar.selectbox(
        "LLM Provider",
        provider_options,
        index=current_idx,
        help="Select the LLM provider. '(default)' uses the LLM_PROVIDER env var.",
    )
    st.session_state.provider = (
        None if selected_provider == "(default)" else selected_provider
    )

    # Provider info
    if available_providers:
        all_info = get_provider_info()
        with st.sidebar.expander("Provider details"):
            for pname, pinfo in all_info.items():
                status = "configured" if pinfo.get("configured") else "not configured"
                st.markdown(
                    f"- **{pname}**: {status} (model: `{pinfo.get('model', 'N/A')}`)"
                )

    st.session_state.model = st.sidebar.text_input(
        "Model",
        value=st.session_state.model,
        help="LLM model name (e.g. qwen3-vl).",
    )

    st.sidebar.subheader("Pipeline Tuning")
    st.session_state.max_sub_queries = st.sidebar.slider(
        "Max sub-queries",
        1,
        10,
        st.session_state.max_sub_queries,
    )
    st.session_state.min_sub_queries = st.sidebar.slider(
        "Min sub-queries",
        1,
        10,
        st.session_state.min_sub_queries,
    )
    st.session_state.max_grep_results = st.sidebar.slider(
        "Max grep results",
        5,
        100,
        st.session_state.max_grep_results,
    )
    st.session_state.max_total_chars = st.sidebar.slider(
        "Max total chars (merger)",
        1000,
        50_000,
        st.session_state.max_total_chars,
        step=1000,
    )
    st.session_state.max_chunks = st.sidebar.slider(
        "Max chunks (merger)",
        5,
        100,
        st.session_state.max_chunks,
    )
    st.session_state.max_chars_per_node = st.sidebar.slider(
        "Max chars per node",
        500,
        10_000,
        st.session_state.max_chars_per_node,
        step=500,
    )
    st.session_state.max_context_tokens = st.sidebar.slider(
        "Max context tokens (generator)",
        500,
        10_000,
        st.session_state.max_context_tokens,
        step=500,
    )

    # Rebuild pipeline button
    st.sidebar.divider()
    if st.sidebar.button("Apply Config & Rebuild Pipeline", use_container_width=True):
        _build_pipeline()


def _build_config() -> LDRSConfig:
    """Build an LDRSConfig from current session state."""
    md_dir = st.session_state.md_dir.strip() or None
    return LDRSConfig(
        results_dir=st.session_state.results_dir,
        pdf_dir=st.session_state.pdf_dir,
        md_dir=md_dir,
        model=st.session_state.model,
        provider=st.session_state.provider,
        max_sub_queries=st.session_state.max_sub_queries,
        min_sub_queries=st.session_state.min_sub_queries,
        max_grep_results=st.session_state.max_grep_results,
        max_total_chars=st.session_state.max_total_chars,
        max_chunks=st.session_state.max_chunks,
        max_chars_per_node=st.session_state.max_chars_per_node,
        max_context_tokens=st.session_state.max_context_tokens,
    )


def _build_pipeline():
    """Build and initialise the pipeline with current config."""
    config = _build_config()
    try:
        pipeline = LDRSPipeline(config)
        with st.spinner("Building corpus..."):
            count = pipeline.build_corpus()
        st.session_state.pipeline = pipeline
        st.session_state.corpus_built = True
        st.sidebar.success(f"Pipeline ready: {count} documents")
    except Exception as e:
        st.sidebar.error(f"Pipeline build failed: {e}")
        logger.exception("Pipeline build failed")


# ---------------------------------------------------------------------------
# Tab 1: Query Interface
# ---------------------------------------------------------------------------
def render_query_tab():
    """Render the main query interface."""
    st.header("Query")

    # Auto-build pipeline if not yet done
    if st.session_state.pipeline is None:
        st.info(
            "Pipeline not initialised. Click **Apply Config & Rebuild Pipeline** "
            "in the sidebar, or enter a query below to auto-initialise."
        )

    # Query input
    query_text = st.text_area(
        "Enter your question:",
        height=80,
        placeholder="e.g. What is Earth Mover's Distance?",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        run_query = st.button("Run Query", type="primary", use_container_width=True)
    with col2:
        show_raw = st.checkbox("Show raw JSON result", value=False)

    if run_query and query_text.strip():
        # Auto-init pipeline if needed
        if st.session_state.pipeline is None:
            _build_pipeline()
            if st.session_state.pipeline is None:
                st.error("Could not initialise pipeline. Check config in sidebar.")
                return

        pipeline = st.session_state.pipeline

        # Run query with progress
        with st.spinner("Running LDRS v2 pipeline (LLM calls in progress)..."):
            t0 = time.monotonic()
            try:
                result = run_async(pipeline.query(query_text.strip()))
            except Exception as e:
                st.error(f"Query failed: {e}")
                return
            elapsed = time.monotonic() - t0

        # Store in history
        st.session_state.query_history.insert(0, (query_text.strip(), result))

        # Display result
        _render_result(result, elapsed, show_raw)

    elif run_query and not query_text.strip():
        st.warning("Please enter a query.")

    # Previous results
    if st.session_state.query_history:
        st.divider()
        st.subheader("Query History")
        for i, (q, r) in enumerate(st.session_state.query_history):
            with st.expander(
                f"**{q}**" + (" [ERROR]" if r.error else ""), expanded=(i == 0)
            ):
                _render_result(r, sum(r.timings.values()), show_raw=False)


def _render_result(result: LDRSResult, elapsed: float, show_raw: bool = False):
    """Render a single query result."""
    # Error banner
    if result.error:
        st.error(f"Pipeline error: {result.error}")

    # Answer
    st.markdown("### Answer")
    st.markdown(result.answer)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sub-queries", len(result.sub_queries))
    col2.metric("Docs selected", len(result.selected_docs))
    col3.metric("Grep hits", result.grep_hits)
    col4.metric("Total time", f"{elapsed:.2f}s")

    # Detailed sections in tabs
    detail_tabs = st.tabs(
        [
            "Citations",
            "Sub-queries",
            "Selected Docs",
            "Context Stats",
            "Timings",
            "Reasoning",
        ]
    )

    # --- Citations ---
    with detail_tabs[0]:
        if result.citations:
            for c in result.citations:
                st.markdown(
                    f"- **[{c['node_id']}]** {c['section']} "
                    f"(Pages {c['pages']}) — `{c['doc_name']}`"
                )
        else:
            st.info("No citations generated.")

    # --- Sub-queries ---
    with detail_tabs[1]:
        if result.sub_queries:
            for i, sq in enumerate(result.sub_queries, 1):
                st.markdown(f"{i}. {sq}")
        else:
            st.info("No sub-queries generated.")

    # --- Selected docs ---
    with detail_tabs[2]:
        if result.selected_docs:
            for doc in result.selected_docs:
                st.markdown(f"- `{doc}`")
        else:
            st.info("No documents selected.")

    # --- Context stats ---
    with detail_tabs[3]:
        if result.merged_context:
            mc = result.merged_context
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Chunks", mc.num_chunks)
            mcol2.metric("Total chars", f"{mc.total_chars:,}")
            mcol3.metric("Documents", mc.num_docs)
            mcol4.metric("Dropped", mc.dropped_count)

            if mc.chunks:
                st.markdown("**Chunks breakdown:**")
                for chunk in mc.chunks:
                    start, end = chunk.page_range
                    st.markdown(
                        f"- **{chunk.title}** (`{chunk.doc_name}`, "
                        f"node `{chunk.node_id}`, pages {start}-{end}, "
                        f"score {chunk.relevance_score:.1f}, "
                        f"{len(chunk.text):,} chars)"
                    )
        else:
            st.info("No merged context available.")

    # --- Timings ---
    with detail_tabs[4]:
        if result.timings:
            # Bar chart via simple columns
            for stage, secs in result.timings.items():
                col_a, col_b = st.columns([3, 1])
                col_a.progress(
                    min(secs / max(elapsed, 0.01), 1.0),
                    text=stage,
                )
                col_b.write(f"{secs:.2f}s")
            st.markdown(f"**Total: {elapsed:.2f}s**")
        else:
            st.info("No timing data available.")

    # --- Reasoning ---
    with detail_tabs[5]:
        if result.expansion_reasoning:
            st.markdown("**Query Expansion Reasoning:**")
            st.text(result.expansion_reasoning)
        if result.selection_reasoning:
            st.markdown("**Document Selection Reasoning:**")
            st.text(result.selection_reasoning)
        if not result.expansion_reasoning and not result.selection_reasoning:
            st.info("No reasoning data available.")

    # Raw JSON
    if show_raw:
        st.divider()
        st.markdown("### Raw Result (JSON)")
        raw = {
            "query": result.query,
            "answer": result.answer,
            "sub_queries": result.sub_queries,
            "selected_docs": result.selected_docs,
            "grep_hits": result.grep_hits,
            "citations": result.citations,
            "expansion_reasoning": result.expansion_reasoning,
            "selection_reasoning": result.selection_reasoning,
            "timings": result.timings,
            "error": result.error,
        }
        if result.merged_context:
            raw["merged_context"] = {
                "num_chunks": result.merged_context.num_chunks,
                "total_chars": result.merged_context.total_chars,
                "num_docs": result.merged_context.num_docs,
                "dropped_count": result.merged_context.dropped_count,
            }
        st.json(raw)


# ---------------------------------------------------------------------------
# Tab 2: Corpus Management
# ---------------------------------------------------------------------------
def render_corpus_tab():
    """Render the corpus management interface."""
    st.header("Corpus Management")

    if st.session_state.pipeline is None:
        st.warning("Pipeline not initialised. Use the sidebar to build it first.")
        return

    pipeline = st.session_state.pipeline

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Corpus Stats", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Rebuild Corpus", type="primary", use_container_width=True):
            with st.spinner("Rebuilding corpus..."):
                count = pipeline.build_corpus()
            st.success(f"Corpus rebuilt: {count} documents")
            st.rerun()

    # Stats
    stats = pipeline.corpus_stats()

    st.subheader("Overview")
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric("Documents", stats["num_documents"])
    scol2.metric("Changelog entries", stats["changelog_entries"])
    scol3.metric("Active docs", len(stats.get("active_docs", [])))

    # Document list with details
    st.subheader("Registered Documents")
    if stats["doc_names"]:
        for doc_name in sorted(stats["doc_names"]):
            entry = pipeline.registry.get_entry(doc_name)
            if entry:
                with st.expander(f"`{doc_name}`"):
                    ecol1, ecol2 = st.columns(2)
                    ecol1.markdown(f"**Node count:** {entry.get('node_count', 'N/A')}")
                    ecol2.markdown(f"**Page range:** {entry.get('page_range', 'N/A')}")

                    desc = entry.get("doc_description", "")
                    if desc:
                        st.markdown(f"**Description:** {desc}")

                    sections = entry.get("top_level_sections", [])
                    if sections:
                        st.markdown("**Top-level sections:**")
                        for sec in sections:
                            st.markdown(f"- {sec}")

                    st.markdown(f"**Index:** `{entry.get('index_path', 'N/A')}`")
                    md_path = entry.get("md_path", "")
                    if md_path:
                        st.markdown(f"**Markdown:** `{md_path}`")
                    st.markdown(f"**Indexed at:** {entry.get('indexed_at', 'N/A')}")
    else:
        st.info("No documents in corpus.")

    # Corpus summary (LLM-ready)
    st.subheader("Corpus Summary (LLM-ready)")
    summary = pipeline.corpus_summary()
    st.text_area("Summary text", value=summary, height=200, disabled=True)

    # Raw stats JSON
    with st.expander("Raw stats JSON"):
        st.json(stats)


# ---------------------------------------------------------------------------
# Tab 3: Document Indexing
# ---------------------------------------------------------------------------
def render_indexing_tab():
    """Render the document indexing interface."""
    st.header("Index a Document")

    if st.session_state.pipeline is None:
        st.warning("Pipeline not initialised. Use the sidebar to build it first.")
        return

    st.markdown(
        "Index a new PDF into the corpus. You need both the **PDF file** and its "
        "**PageIndex structure JSON** file."
    )

    # File paths
    pdf_path = st.text_input(
        "PDF file path",
        placeholder="/path/to/document.pdf",
        help="Absolute or relative path to the PDF file.",
    )
    index_path = st.text_input(
        "Structure JSON path",
        placeholder="/path/to/document_structure.json",
        help="Absolute or relative path to the *_structure.json file.",
    )
    md_filename = st.text_input(
        "Custom .md filename (optional)",
        placeholder="Leave blank for auto-naming",
        help="If blank, the doc_name from the index is used.",
    )

    use_ocr = st.checkbox(
        "Use OCR for text extraction (Recommended for Nepali text)",
        value=False,
        help="Extract text using OCR rather than standard font-based extraction.",
        key="index_use_ocr",
    )

    if st.button("Index Document", type="primary"):
        if not pdf_path.strip() or not index_path.strip():
            st.error("Both PDF path and structure JSON path are required.")
            return

        # Validate files exist
        if not os.path.exists(pdf_path.strip()):
            st.error(f"PDF file not found: {pdf_path}")
            return
        if not os.path.exists(index_path.strip()):
            st.error(f"Structure JSON not found: {index_path}")
            return

        md_fn = md_filename.strip() or None
        pipeline = st.session_state.pipeline

        progress_text = "Indexing document... Initializing..."
        progress_bar = st.progress(0, text=progress_text)

        def update_progress_index(current: int, total: int, message: str):
            pct = 0.0 if total == 0 else min(max(current / total, 0.0), 1.0)
            progress_bar.progress(pct, text=f"Indexing: {message}")

        try:
            md_path = pipeline.index_document(
                pdf_path=pdf_path.strip(),
                index_path=index_path.strip(),
                md_filename=md_fn,
                use_ocr=use_ocr,
                progress_callback=update_progress_index,
            )
            progress_bar.progress(1.0, text="Indexing Complete!")
            st.success(f"Document indexed successfully. Markdown: `{md_path}`")
        except Exception as e:
            progress_bar.empty()
            st.error(f"Indexing failed: {e}")
            logger.exception("Document indexing failed")

    # Quick-index from test data
    st.divider()
    st.subheader("Quick Index (Test Data)")
    st.markdown("Select a test document to index from the project's test fixtures:")

    results_dir = st.session_state.results_dir
    pdf_dir = st.session_state.pdf_dir

    if os.path.isdir(results_dir):
        json_files = sorted(
            [
                f
                for f in os.listdir(results_dir)
                if f.endswith("_structure.json") and not f.startswith("_")
            ]
        )
        if json_files:
            selected = st.selectbox("Test document", json_files)
            if selected and st.button("Quick Index Selected", use_container_width=True):
                idx = os.path.join(results_dir, selected)
                # Derive PDF name
                stem = selected.replace("_structure.json", "")
                pdf = os.path.join(pdf_dir, stem + ".pdf")
                if not os.path.exists(pdf):
                    st.error(f"PDF not found: {pdf}")
                else:
                    progress_bar = st.progress(0, text=f"Indexing {stem}...")

                    def update_progress_quick(current: int, total: int, message: str):
                        pct = 0.0 if total == 0 else min(max(current / total, 0.0), 1.0)
                        progress_bar.progress(pct, text=f"{stem}: {message}")

                    try:
                        md_path = st.session_state.pipeline.index_document(
                            pdf_path=pdf,
                            index_path=idx,
                            progress_callback=update_progress_quick,
                        )
                        progress_bar.progress(1.0, text="Complete!")
                        st.success(f"Indexed: `{md_path}`")
                    except Exception as e:
                        progress_bar.empty()
                        st.error(f"Failed: {e}")
        else:
            st.info(f"No *_structure.json files found in {results_dir}")
    else:
        st.warning(f"Results directory does not exist: {results_dir}")


# ---------------------------------------------------------------------------
# Tab 4: PageIndex (PDF → Structure → Register)
# ---------------------------------------------------------------------------
def render_pageindex_tab():
    """Render the PageIndex interface for indexing raw PDFs."""
    st.header("PageIndex — Index from Raw PDF")

    if st.session_state.pipeline is None:
        st.warning("Pipeline not initialised. Use the sidebar to build it first.")
        return

    st.markdown(
        "Run **PageIndex** on a raw PDF to automatically generate a structure JSON, "
        "extract markdown, and register the document in the corpus. "
        "This is the full pipeline from a raw PDF to a corpus-ready document."
    )

    # File path input
    pdf_path = st.text_input(
        "PDF file path",
        placeholder="/path/to/document.pdf",
        help="Absolute or relative path to the PDF file.",
        key="pageindex_pdf_path",
    )
    output_dir = st.text_input(
        "Output directory (optional)",
        placeholder=f"Defaults to: {st.session_state.results_dir}",
        help="Directory to save the structure JSON and .md files.",
        key="pageindex_output_dir",
    )
    md_filename = st.text_input(
        "Custom .md filename (optional)",
        placeholder="Leave blank for auto-naming",
        help="If blank, the doc_name from the generated index is used.",
        key="pageindex_md_filename",
    )

    use_ocr = st.checkbox(
        "Use OCR for text extraction (Recommended for Nepali text)",
        value=False,
        help="Extract text using OCR rather than standard font-based extraction. Essential for documents containing Devanagari script.",
        key="pageindex_use_ocr",
    )

    if st.button("Run PageIndex", type="primary", key="run_pageindex"):
        if not pdf_path.strip():
            st.error("PDF file path is required.")
            return

        if not os.path.exists(pdf_path.strip()):
            st.error(f"PDF file not found: {pdf_path}")
            return

        out_dir = output_dir.strip() or None
        md_fn = md_filename.strip() or None
        pipeline = st.session_state.pipeline

        # Set up a progress bar
        progress_text = "Running PageIndex... Initializing..."
        progress_bar = st.progress(0, text=progress_text)

        def update_progress(current: int, total: int, message: str):
            # Calculate percentage (0.0 to 1.0)
            pct = 0.0 if total == 0 else min(max(current / total, 0.0), 1.0)
            progress_bar.progress(pct, text=f"PageIndex: {message}")

        try:
            md_path = run_async(
                pipeline.index_document_from_pdf(
                    pdf_path=pdf_path.strip(),
                    output_dir=out_dir,
                    md_filename=md_fn,
                    use_ocr=use_ocr,
                    progress_callback=update_progress,
                )
            )
            progress_bar.progress(1.0, text="PageIndex Complete!")
            st.success(
                f"Document indexed successfully!\n\n"
                f"Markdown file: `{md_path}`\n\n"
                f"Corpus now has "
                f"**{len(pipeline.registry.doc_names)}** documents."
            )
        except Exception as e:
            progress_bar.empty()
            st.error(f"PageIndex failed: {e}")
            logger.exception("PageIndex indexing failed")

    # Quick-index from Nepalidocs
    st.divider()
    st.subheader("Quick Index — Nepali Documents")
    st.markdown("Select a PDF from the `Nepalidocs/` folder to run PageIndex on it.")

    nepali_dir = os.path.join(PROJECT_ROOT, "Nepalidocs")
    if os.path.isdir(nepali_dir):
        pdf_files = sorted(
            [f for f in os.listdir(nepali_dir) if f.lower().endswith(".pdf")]
        )
        if pdf_files:
            selected = st.selectbox("Nepali PDF", pdf_files, key="nepali_pdf_select")
            use_ocr_nepali = st.checkbox("Use OCR", value=True, key="use_ocr_nepali")
            if selected and st.button(
                "Run PageIndex on Selected",
                use_container_width=True,
                key="run_pageindex_nepali",
            ):
                full_path = os.path.join(nepali_dir, selected)
                out_dir = os.path.join(nepali_dir, "index")
                os.makedirs(out_dir, exist_ok=True)

                progress_bar = st.progress(
                    0, text=f"Running PageIndex on {selected}..."
                )

                def update_progress_nepali(current: int, total: int, message: str):
                    pct = 0.0 if total == 0 else min(max(current / total, 0.0), 1.0)
                    progress_bar.progress(pct, text=f"{selected}: {message}")

                try:
                    md_path = run_async(
                        st.session_state.pipeline.index_document_from_pdf(
                            pdf_path=full_path,
                            output_dir=out_dir,
                            use_ocr=use_ocr_nepali,
                            progress_callback=update_progress_nepali,
                        )
                    )
                    progress_bar.progress(1.0, text="Complete!")
                    st.success(f"Indexed: `{md_path}`")
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Failed: {e}")
                    logger.exception("Nepali PageIndex failed")
        else:
            st.info("No PDF files found in Nepalidocs/")
    else:
        st.info("Nepalidocs/ directory not found.")


# ---------------------------------------------------------------------------
# Tab 5: Batch Query
# ---------------------------------------------------------------------------
def render_batch_tab():
    """Render the batch query interface."""
    st.header("Batch Query")

    if st.session_state.pipeline is None:
        st.warning("Pipeline not initialised. Use the sidebar to build it first.")
        return

    st.markdown("Run multiple queries in sequence. Enter one query per line.")

    queries_text = st.text_area(
        "Queries (one per line):",
        height=150,
        placeholder="What is Earth Mover's Distance?\nExplain pattern recognition.\nWhat are the key financial results?",
    )

    if st.button("Run Batch", type="primary"):
        lines = [q.strip() for q in queries_text.strip().split("\n") if q.strip()]
        if not lines:
            st.warning("No queries entered.")
            return

        pipeline = st.session_state.pipeline
        st.markdown(f"Running **{len(lines)}** queries...")

        progress_bar = st.progress(0, text="Starting batch...")
        results = []

        for i, q in enumerate(lines):
            progress_bar.progress(
                (i) / len(lines),
                text=f"Query {i + 1}/{len(lines)}: {q[:60]}...",
            )
            try:
                r = run_async(pipeline.query(q))
                results.append((q, r))
                st.session_state.query_history.insert(0, (q, r))
            except Exception as e:
                st.error(f"Query {i + 1} failed: {e}")
                # Create an error result
                err_result = LDRSResult(query=q, error=str(e), answer=f"Error: {e}")
                results.append((q, err_result))

        progress_bar.progress(1.0, text="Batch complete!")

        # Display results
        for i, (q, r) in enumerate(results):
            with st.expander(
                f"**Q{i + 1}:** {q}" + (" [ERROR]" if r.error else ""),
                expanded=(i == 0),
            ):
                _render_result(r, sum(r.timings.values()), show_raw=False)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
def main():
    """Main Streamlit app entry point."""
    render_sidebar()

    st.title("LDRS v2 — Living Document RAG System")
    st.caption(
        "Filesystem-native, tree-aware document retrieval with LLM-powered "
        "query expansion, document selection, and answer generation."
    )

    # Status bar
    if st.session_state.pipeline is not None and st.session_state.corpus_built:
        stats = st.session_state.pipeline.corpus_stats()
        provider_label = st.session_state.pipeline.llm_provider.provider_name
        st.success(
            f"Pipeline active | Provider: `{provider_label}` | "
            f"Model: `{st.session_state.model}` | "
            f"Corpus: {stats['num_documents']} documents"
        )
    else:
        st.warning(
            "Pipeline not active. Configure directories in the sidebar and click "
            "**Apply Config & Rebuild Pipeline**."
        )

    # Main tabs
    tab_query, tab_corpus, tab_index, tab_pageindex, tab_batch = st.tabs(
        [
            "Query",
            "Corpus",
            "Index Document",
            "PageIndex (PDF)",
            "Batch Query",
        ]
    )

    with tab_query:
        render_query_tab()

    with tab_corpus:
        render_corpus_tab()

    with tab_index:
        render_indexing_tab()

    with tab_pageindex:
        render_pageindex_tab()

    with tab_batch:
        render_batch_tab()


if __name__ == "__main__":
    main()
