"""
LDRS v2 Pipeline — End-to-end query orchestration.

This module wires together all v2 components into a single coherent pipeline:

    Query
      → QueryExpander   (LLM: 3-5 sub-queries)
      → DocSelector     (LLM: pick relevant docs from corpus)
      → Parallel per-doc:
          TreeGrep      (title/summary/body search)
          ContextFetcher (fetch .md body for matched nodes)
      → ContextMerger   (rank, dedup, budget across all docs)
      → Generator       (LLM: answer with citations)
      → LDRSResult

The pipeline also provides an ``index_document`` method for ingesting
new PDFs into the corpus (extract .md, register in DocRegistry, log
to ChangeLog).

Configuration is centralised in :class:`LDRSConfig`.

Usage::

    config = LDRSConfig(
        results_dir="tests/results",
        pdf_dir="tests/pdfs",
        md_dir="tests/results",
    )
    pipeline = LDRSPipeline(config)
    pipeline.build_corpus()

    result = await pipeline.query("What is Earth Mover's Distance?")
    print(result.answer)
    print(result.citations)

Architecture notes:
  - All LLM calls are async (``asyncio``).
  - Per-document retrieval is parallelised with ``asyncio.gather``.
  - Every stage is individually configurable (model, budgets, caps).
  - Verbose ``logging.DEBUG`` throughout.
  - UTF-8 NFC normalization is enforced at every boundary.
"""

import asyncio
import json
import logging
import os
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ldrs.changelog import ChangeLog
from ldrs.context_merger import ContextChunk, ContextMerger, MergedContext
from ldrs.doc_registry import DocRegistry, build_entry
from ldrs.doc_selector import DocSelector, DocSelection
from ldrs.llm_provider import LLMProvider, get_provider
from ldrs.pdf_extractor import (
    PdfExtractor,
    build_line_to_page_map,
    map_structure_pages,
)
from ldrs.query_expander import ExpandedQuery, QueryExpander
from ldrs.tree_grep import TreeGrep, GrepResult
from rag.context_fetcher import ContextFetcher
from rag.generator import Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LDRSConfig:
    """
    Centralised configuration for the LDRS v2 pipeline.

    All directories, model names, and tuning parameters live here so that
    the pipeline class itself stays clean.

    Attributes:
        results_dir:     Directory containing ``*_structure.json`` index files.
        pdf_dir:         Directory containing source PDF files.
        md_dir:          Directory for cached ``.md`` files (defaults to
                         ``results_dir`` if not specified).
        registry_path:   Path to the ``_registry.json`` file (auto-generated).
        changelog_path:  Path to the ``_changelog.json`` file (auto-generated).
        provider:        LLM provider name ("local", "openai", "gemini").
                         If None, reads from ``LLM_PROVIDER`` env var.
        model:           Default LLM model name for all LLM-calling stages.
                         If None, the provider's default model is used.
        max_sub_queries: Max sub-queries from QueryExpander.
        min_sub_queries: Min sub-queries from QueryExpander.
        max_grep_results:  Max results per TreeGrep search.
        max_total_chars:   Character budget for ContextMerger.
        max_chunks:        Chunk count cap for ContextMerger.
        max_chars_per_node: Per-node character cap for ContextFetcher.
        max_context_tokens: Token budget for Generator.
    """

    # Directories
    results_dir: str = "tests/results"
    pdf_dir: str = "tests/pdfs"
    md_dir: Optional[str] = None  # defaults to results_dir

    # Persistence paths (auto-set if None)
    registry_path: Optional[str] = None
    changelog_path: Optional[str] = None

    # LLM settings
    provider: Optional[str] = None  # None → reads LLM_PROVIDER env var
    model: str = "qwen3-vl"

    # QueryExpander tuning
    max_sub_queries: int = 5
    min_sub_queries: int = 3

    # TreeGrep tuning
    max_grep_results: int = 30

    # ContextMerger tuning
    max_total_chars: int = 15_000
    max_chunks: int = 30

    # ContextFetcher tuning
    max_chars_per_node: int = 4000

    # Generator tuning
    max_context_tokens: int = 4000

    def __post_init__(self):
        """Set derived defaults."""
        if self.md_dir is None:
            self.md_dir = self.results_dir
        if self.registry_path is None:
            self.registry_path = os.path.join(self.results_dir, "_registry.json")
        if self.changelog_path is None:
            self.changelog_path = os.path.join(self.results_dir, "_changelog.json")


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------


@dataclass
class LDRSResult:
    """
    Complete result of an LDRS v2 query.

    Attributes:
        query:              The original user query.
        answer:             Generated answer text.
        sub_queries:        Expanded sub-queries used.
        selected_docs:      Documents chosen by DocSelector.
        grep_hits:          Total TreeGrep hits across all docs.
        merged_context:     The :class:`MergedContext` from ContextMerger.
        citations:          List of citation dicts (doc, section, pages).
        expansion_reasoning: Why QueryExpander chose these sub-queries.
        selection_reasoning: Why DocSelector chose these documents.
        timings:            Dict of stage-name → elapsed seconds.
        error:              Error message if the pipeline failed.
    """

    query: str = ""
    answer: str = ""
    sub_queries: List[str] = field(default_factory=list)
    selected_docs: List[str] = field(default_factory=list)
    grep_hits: int = 0
    merged_context: Optional[MergedContext] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    expansion_reasoning: str = ""
    selection_reasoning: str = ""
    timings: Dict[str, float] = field(default_factory=dict)
    error: str = ""


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class LDRSPipeline:
    """
    End-to-end LDRS v2 pipeline.

    Lifecycle:

    1. Instantiate with :class:`LDRSConfig`.
    2. Call :meth:`build_corpus` to scan the results directory and populate
       the DocRegistry and ChangeLog.
    3. Call :meth:`query` (async) to run the full retrieval + generation
       pipeline.

    Alternatively, use :meth:`index_document` to add individual PDFs.

    Args:
        config: An :class:`LDRSConfig` instance.

    Example::

        config = LDRSConfig(results_dir="tests/results", pdf_dir="tests/pdfs")
        pipeline = LDRSPipeline(config)
        pipeline.build_corpus()

        result = await pipeline.query("What is the EMD algorithm?")
        print(result.answer)
    """

    def __init__(self, config: LDRSConfig):
        self.config = config
        logger.info(
            "LDRSPipeline init  results_dir=%s  pdf_dir=%s  provider=%s  model=%s",
            config.results_dir,
            config.pdf_dir,
            config.provider or "(env default)",
            config.model,
        )

        # Centralised LLM provider — shared by all LLM-calling components
        self.llm_provider: LLMProvider = get_provider(
            provider_name=config.provider,
            model_override=config.model,
        )

        # Core components
        self.registry = DocRegistry(registry_path=config.registry_path)
        self.changelog = ChangeLog(changelog_path=config.changelog_path)

        # LLM-calling stages (lazy — only created when needed)
        self._query_expander: Optional[QueryExpander] = None
        self._doc_selector: Optional[DocSelector] = None
        self._generator: Optional[Generator] = None

    # ------------------------------------------------------------------
    # Lazy component initialisation
    # ------------------------------------------------------------------

    @property
    def query_expander(self) -> QueryExpander:
        """Lazy-init QueryExpander."""
        if self._query_expander is None:
            self._query_expander = QueryExpander(
                model=self.config.model,
                max_sub_queries=self.config.max_sub_queries,
                min_sub_queries=self.config.min_sub_queries,
                llm_provider=self.llm_provider,
            )
        return self._query_expander

    @property
    def doc_selector(self) -> DocSelector:
        """Lazy-init DocSelector."""
        if self._doc_selector is None:
            self._doc_selector = DocSelector(
                model=self.config.model,
                llm_provider=self.llm_provider,
            )
        return self._doc_selector

    @property
    def generator(self) -> Generator:
        """Lazy-init Generator."""
        if self._generator is None:
            self._generator = Generator(
                model=self.config.model,
                max_context_tokens=self.config.max_context_tokens,
                llm_provider=self.llm_provider,
            )
        return self._generator

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def build_corpus(self) -> int:
        """
        Scan the results directory and rebuild the registry + changelog.

        Scans for ``*_structure.json`` files, builds a registry entry for
        each, auto-generates missing ``.md`` files from PDFs using
        :class:`PdfExtractor` (font-based heading detection, no LLM calls),
        and records all documents in the changelog.

        Returns:
            Number of documents registered.
        """
        logger.info(
            "LDRSPipeline.build_corpus  results_dir=%s  md_dir=%s  pdf_dir=%s",
            self.config.results_dir,
            self.config.md_dir,
            self.config.pdf_dir,
        )

        t0 = time.monotonic()
        count = self.registry.rebuild(
            results_dir=self.config.results_dir,
            md_dir=self.config.md_dir,
        )

        # ------------------------------------------------------------------
        # Auto-generate missing .md files from PDFs using PdfExtractor
        # ------------------------------------------------------------------
        md_generated = 0
        for doc_name in self.registry.doc_names:
            entry = self.registry.get_entry(doc_name)
            if not entry:
                continue

            md_path = entry.get("md_path")
            if md_path and os.path.exists(md_path):
                continue  # .md already exists, nothing to do

            idx_path = entry.get("index_path", "")
            pdf_path = self._find_pdf_path(doc_name)
            if not pdf_path:
                logger.warning(
                    "build_corpus: no PDF found for %s — cannot extract .md",
                    doc_name,
                )
                continue

            # Extract .md from PDF using PdfExtractor (no structure JSON needed)
            try:
                extractor = PdfExtractor(
                    pdf_path=pdf_path,
                    output_dir=self.config.md_dir,
                )
                new_md_path = extractor.extract()
                logger.info(
                    "build_corpus: extracted .md for %s → %s",
                    doc_name,
                    new_md_path,
                )

                # Update the registry entry with the new md_path
                self.registry.add_or_update(idx_path, new_md_path)
                md_generated += 1

            except Exception as exc:
                logger.error(
                    "build_corpus: failed to extract .md for %s: %s",
                    doc_name,
                    exc,
                )

        if md_generated:
            logger.info(
                "build_corpus: auto-generated %d .md files",
                md_generated,
            )

        self.registry.save()

        # Record each document in the changelog
        for doc_name in self.registry.doc_names:
            entry = self.registry.get_entry(doc_name)
            if entry:
                idx_path = entry.get("index_path", "")
                # Load the full index JSON so changelog can compute commit_id
                try:
                    with open(idx_path, "r", encoding="utf-8") as f:
                        index_data = json.load(f)
                    structure = index_data.get("structure", [])
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "build_corpus: could not load index for %s: %s",
                        doc_name,
                        exc,
                    )
                    index_data = {}
                    structure = []
                self.changelog.record_indexed(
                    doc_name=doc_name,
                    index_data=index_data,
                    structure=structure,
                )
        self.changelog.save()

        elapsed = time.monotonic() - t0
        logger.info(
            "LDRSPipeline.build_corpus  done  docs=%d  md_generated=%d  elapsed=%.2fs",
            count,
            md_generated,
            elapsed,
        )
        return count

    def index_document(
        self,
        pdf_path: str,
        index_path: str,
        md_filename: Optional[str] = None,
    ) -> str:
        """
        Index a single document: extract .md, register, and log.

        Uses :class:`PdfExtractor` (font-based heading detection) to
        extract markdown from the PDF.  The ``index_path`` (structure JSON)
        is used for the DocRegistry and ChangeLog only — it is NOT needed
        for the markdown extraction step.

        Args:
            pdf_path:    Path to the source PDF file.
            index_path:  Path to the ``*_structure.json`` file.
            md_filename: Custom ``.md`` filename.  If ``None``, uses the
                         document name from the index.

        Returns:
            Path to the generated ``.md`` file.
        """
        logger.info(
            "LDRSPipeline.index_document  pdf=%s  index=%s", pdf_path, index_path
        )

        # Extract markdown using PdfExtractor (no structure JSON needed)
        extractor = PdfExtractor(
            pdf_path=pdf_path,
            output_dir=self.config.md_dir,
        )
        md_path = extractor.extract(output_filename=md_filename)
        logger.info("LDRSPipeline.index_document  md extracted: %s", md_path)

        # Register
        entry = self.registry.add_or_update(index_path, md_path)
        self.registry.save()

        # Log — load full index JSON for changelog commit hash
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        structure = index_data.get("structure", [])
        doc_name = entry.get("doc_name", os.path.basename(pdf_path))
        self.changelog.record_indexed(
            doc_name=doc_name,
            index_data=index_data,
            structure=structure,
        )
        self.changelog.save()

        logger.info(
            "LDRSPipeline.index_document  done  doc=%s  nodes=%d",
            doc_name,
            entry.get("node_count", 0),
        )
        return md_path

    async def index_document_from_pdf(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        md_filename: Optional[str] = None,
        if_add_node_summary: str = "no",
        if_add_doc_description: str = "no",
    ) -> str:
        """
        Generate a structure index from a raw PDF and ingest it into the corpus.

        New flow (no heavy LLM-based PageIndex):

        1. Run :class:`PdfExtractor` to extract full document text into
           Markdown with font-based heading detection and ``<!-- page: N -->``
           markers.
        2. Run ``pageindex.page_index_md.md_to_tree()`` to parse the
           Markdown headings into a hierarchical structure tree.
        3. Run :func:`map_structure_pages` to convert ``line_num`` fields
           into ``start_index`` / ``end_index`` (PDF page numbers) for
           backward compatibility with TreeGrep, ContextFetcher, etc.
        4. Save the structure JSON to ``{output_dir}/{stem}_structure.json``.
        5. Register in DocRegistry and log to ChangeLog.

        This replaces the old flow that used ``pageindex.page_index()``
        (which required 20+ LLM calls for TOC detection, verification, etc.).

        Args:
            pdf_path:              Path to the source PDF file.
            output_dir:            Directory to save outputs.
                                   Defaults to ``config.results_dir``.
            md_filename:           Custom ``.md`` filename.
            if_add_node_summary:   "yes" to generate LLM summaries for nodes.
            if_add_doc_description: "yes" to generate an LLM doc description.

        Returns:
            Path to the generated ``.md`` file.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ValueError: If md_to_tree fails to produce a valid structure.
        """
        import unicodedata as _ud

        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        output_dir = output_dir or self.config.results_dir
        md_output_dir = self.config.md_dir or output_dir
        logger.info(
            "LDRSPipeline.index_document_from_pdf  pdf=%s  output_dir=%s",
            pdf_path,
            output_dir,
        )

        # ---- Step 1: Extract PDF → Markdown using PdfExtractor ----
        t0 = time.monotonic()
        extractor = PdfExtractor(
            pdf_path=pdf_path,
            output_dir=md_output_dir,
        )
        md_path = extractor.extract(output_filename=md_filename)
        elapsed_extract = time.monotonic() - t0

        doc_name = _ud.normalize("NFC", os.path.basename(pdf_path))
        stem = os.path.splitext(doc_name)[0]

        logger.info(
            "LDRSPipeline.index_document_from_pdf  md extracted  "
            "doc=%s  md=%s  elapsed=%.1fs",
            doc_name,
            md_path,
            elapsed_extract,
        )

        # ---- Step 2: Parse Markdown → structure tree via md_to_tree ----
        t1 = time.monotonic()
        from pageindex.page_index_md import md_to_tree

        tree_result = await md_to_tree(
            md_path=md_path,
            if_add_node_summary=if_add_node_summary,
            if_add_doc_description=if_add_doc_description,
            if_add_node_id="yes",
            model=self.config.model,
        )
        elapsed_tree = time.monotonic() - t1

        if not tree_result or "structure" not in tree_result:
            raise ValueError(
                f"md_to_tree returned invalid result for {md_path}: {tree_result}"
            )

        structure = tree_result.get("structure", [])
        logger.info(
            "LDRSPipeline.index_document_from_pdf  md_to_tree done  "
            "nodes=%d  elapsed=%.1fs",
            len(structure),
            elapsed_tree,
        )

        # ---- Step 3: Map line_num → start_index / end_index (page numbers) ----
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        line_to_page = build_line_to_page_map(md_text)
        structure = map_structure_pages(structure, line_to_page, md_text)

        logger.info(
            "LDRSPipeline.index_document_from_pdf  page mapping done  "
            "total_line_mappings=%d",
            len(line_to_page),
        )

        # ---- Step 4: Build and save structure JSON ----
        index_data = {
            "doc_name": tree_result.get("doc_name", stem),
        }
        # Include doc_description if available
        if tree_result.get("doc_description"):
            index_data["doc_description"] = tree_result["doc_description"]
        index_data["structure"] = structure

        os.makedirs(output_dir, exist_ok=True)
        index_path = os.path.join(output_dir, f"{stem}_structure.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        logger.info(
            "LDRSPipeline.index_document_from_pdf  saved structure: %s",
            index_path,
        )

        # ---- Step 5: Register in DocRegistry + ChangeLog ----
        entry = self.registry.add_or_update(index_path, md_path)
        self.registry.save()

        reg_doc_name = entry.get("doc_name", doc_name)
        self.changelog.record_indexed(
            doc_name=reg_doc_name,
            index_data=index_data,
            structure=structure,
        )
        self.changelog.save()

        total_elapsed = time.monotonic() - t0
        logger.info(
            "LDRSPipeline.index_document_from_pdf  complete  doc=%s  md=%s  "
            "index=%s  nodes=%d  total=%.1fs",
            reg_doc_name,
            md_path,
            index_path,
            entry.get("node_count", 0),
            total_elapsed,
        )
        return md_path

    # ------------------------------------------------------------------
    # Query pipeline
    # ------------------------------------------------------------------

    async def query(self, user_query: str) -> LDRSResult:
        """
        Run the full LDRS v2 query pipeline.

        Flow::

            user_query
              → QueryExpander     (3-5 sub-queries)
              → DocSelector       (pick relevant docs)
              → Parallel per-doc:
                  TreeGrep        (search titles + summaries + body)
                  ContextFetcher  (fetch .md body for matched nodes)
              → ContextMerger     (rank, dedup, budget)
              → Generator         (LLM answer with citations)
              → LDRSResult

        Args:
            user_query: The natural-language question.

        Returns:
            :class:`LDRSResult` with answer, citations, timings, etc.
            On error, ``result.error`` is populated and ``result.answer``
            contains a user-friendly fallback message.
        """
        logger.info("LDRSPipeline.query  user_query=%r", user_query)
        result = LDRSResult(query=user_query)
        timings: Dict[str, float] = {}

        # NFC-normalize the query
        user_query = unicodedata.normalize("NFC", user_query)

        try:
            # ============================================================
            # Stage 1: Query Expansion
            # ============================================================
            t0 = time.monotonic()
            expanded: ExpandedQuery = await self.query_expander.expand(user_query)
            timings["query_expansion"] = time.monotonic() - t0

            result.sub_queries = expanded.sub_queries
            result.expansion_reasoning = expanded.reasoning
            logger.info(
                "Stage 1 done  sub_queries=%d  elapsed=%.2fs",
                len(expanded.sub_queries),
                timings["query_expansion"],
            )

            # ============================================================
            # Stage 2: Document Selection
            # ============================================================
            t0 = time.monotonic()
            registry_summary = self.registry.to_llm_summary()
            changelog_summary = self.changelog.get_corpus_summary()
            all_doc_names = self.registry.doc_names

            selection: DocSelection = await self.doc_selector.select(
                original_query=user_query,
                sub_queries=expanded.sub_queries,
                registry_summary=registry_summary,
                changelog_summary=changelog_summary,
                all_doc_names=all_doc_names,
            )
            timings["doc_selection"] = time.monotonic() - t0

            result.selected_docs = selection.selected_docs
            result.selection_reasoning = selection.reasoning
            logger.info(
                "Stage 2 done  selected=%d/%d  elapsed=%.2fs",
                len(selection.selected_docs),
                len(all_doc_names),
                timings["doc_selection"],
            )

            if not selection.selected_docs:
                result.answer = (
                    "No documents were selected as relevant to this query. "
                    "The corpus may not contain information about this topic."
                )
                result.timings = timings
                return result

            # ============================================================
            # Stage 3: Parallel per-document retrieval
            # ============================================================
            t0 = time.monotonic()
            retrieval_tasks = []
            for doc_name in selection.selected_docs:
                retrieval_tasks.append(
                    self._retrieve_from_document(
                        doc_name=doc_name,
                        sub_queries=expanded.sub_queries,
                    )
                )

            doc_results: List[
                Tuple[str, List[GrepResult], List[str]]
            ] = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
            timings["retrieval"] = time.monotonic() - t0

            logger.info(
                "Stage 3 done  docs_retrieved=%d  elapsed=%.2fs",
                len(doc_results),
                timings["retrieval"],
            )

            # ============================================================
            # Stage 4: Context merging
            # ============================================================
            t0 = time.monotonic()
            merger = ContextMerger(
                max_total_chars=self.config.max_total_chars,
                max_chunks=self.config.max_chunks,
            )

            total_grep_hits = 0

            for item in doc_results:
                if isinstance(item, Exception):
                    logger.error("Retrieval error: %s", item)
                    continue

                doc_name, grep_hits, context_parts = item
                total_grep_hits += len(grep_hits)

                # Convert grep hits + fetched context into ContextChunks
                self._add_to_merger(
                    merger=merger,
                    doc_name=doc_name,
                    grep_hits=grep_hits,
                    context_parts=context_parts,
                )

            merged: MergedContext = merger.merge()
            timings["merging"] = time.monotonic() - t0

            result.grep_hits = total_grep_hits
            result.merged_context = merged
            logger.info(
                "Stage 4 done  grep_hits=%d  merged_chunks=%d  chars=%d  elapsed=%.2fs",
                total_grep_hits,
                merged.num_chunks,
                merged.total_chars,
                timings["merging"],
            )

            if merged.num_chunks == 0:
                result.answer = (
                    "The selected documents were searched but no relevant "
                    "sections were found matching the query."
                )
                result.timings = timings
                return result

            # ============================================================
            # Stage 5: Answer generation
            # ============================================================
            t0 = time.monotonic()
            # Feed the formatted merged context to the generator as a
            # single-element list (generator expects List[str])
            answer = await self.generator.generate(
                query=user_query,
                context=[merged.formatted_context],
            )
            timings["generation"] = time.monotonic() - t0

            result.answer = answer
            logger.info(
                "Stage 5 done  answer_len=%d  elapsed=%.2fs",
                len(answer),
                timings["generation"],
            )

            # ============================================================
            # Stage 6: Build citations
            # ============================================================
            result.citations = self._build_citations(merged)

        except Exception as e:
            logger.exception("LDRSPipeline.query  pipeline error: %s", e)
            result.error = str(e)
            result.answer = f"An error occurred while processing the query: {e}"

        result.timings = timings
        total_elapsed = sum(timings.values())
        logger.info(
            "LDRSPipeline.query  complete  total=%.2fs  stages=%s",
            total_elapsed,
            {k: f"{v:.2f}s" for k, v in timings.items()},
        )
        return result

    async def batch_query(self, queries: List[str]) -> List[LDRSResult]:
        """
        Run multiple queries sequentially.

        Each query goes through the full pipeline independently.  Sequential
        execution avoids overwhelming the LLM API with parallel requests.

        Args:
            queries: List of user questions.

        Returns:
            List of :class:`LDRSResult` objects (one per query).
        """
        logger.info("LDRSPipeline.batch_query  queries=%d", len(queries))
        results: List[LDRSResult] = []
        for i, q in enumerate(queries, 1):
            logger.info("batch_query  [%d/%d]  query=%r", i, len(queries), q)
            r = await self.query(q)
            results.append(r)
        return results

    # ------------------------------------------------------------------
    # Internal: per-document retrieval
    # ------------------------------------------------------------------

    async def _retrieve_from_document(
        self,
        doc_name: str,
        sub_queries: List[str],
    ) -> Tuple[str, List[GrepResult], List[str]]:
        """
        Retrieve context from a single document.

        Steps:
        1. Look up the document's index_path and md_path from the registry.
        2. Run TreeGrep.search_multi() with all sub-queries.
        3. Collect unique node_ids from grep hits.
        4. Fetch context for those nodes from the .md file.

        Args:
            doc_name:    Document name (as registered).
            sub_queries: Sub-queries from QueryExpander.

        Returns:
            Tuple of (doc_name, grep_hits, context_parts).

        Raises:
            ValueError: If the document is not in the registry.
        """
        logger.info(
            "_retrieve_from_document  doc=%s  sub_queries=%d",
            doc_name,
            len(sub_queries),
        )

        entry = self.registry.get_entry(doc_name)
        if not entry:
            raise ValueError(f"Document not in registry: {doc_name}")

        index_path = entry.get("index_path", "")
        md_path = entry.get("md_path")

        # ------ TreeGrep ------
        grep = TreeGrep(index_path=index_path, md_path=md_path)
        grep_hits = grep.search_multi(
            patterns=sub_queries,
            max_results=self.config.max_grep_results,
        )
        logger.debug(
            "_retrieve_from_document  doc=%s  grep_hits=%d",
            doc_name,
            len(grep_hits),
        )

        if not grep_hits:
            return (doc_name, [], [])

        # ------ Collect unique node_ids (preserve order by relevance) ------
        seen_node_ids: set = set()
        ordered_node_ids: List[str] = []
        for hit in grep_hits:
            if hit.node_id not in seen_node_ids:
                seen_node_ids.add(hit.node_id)
                ordered_node_ids.append(hit.node_id)

        # ------ ContextFetcher ------
        # Determine PDF path for fallback
        pdf_path = self._find_pdf_path(doc_name)
        fetcher = ContextFetcher(index_path=index_path, pdf_path=pdf_path)

        if md_path and os.path.exists(md_path):
            context_parts = fetcher.fetch_from_md(
                node_ids=ordered_node_ids,
                md_path=md_path,
                max_chars_per_node=self.config.max_chars_per_node,
            )
        else:
            # Fallback to PDF
            logger.warning(
                "_retrieve_from_document  no .md for %s, falling back to PDF",
                doc_name,
            )
            context_parts = fetcher.fetch(ordered_node_ids)

        logger.debug(
            "_retrieve_from_document  doc=%s  context_parts=%d",
            doc_name,
            len(context_parts),
        )
        return (doc_name, grep_hits, context_parts)

    def _find_pdf_path(self, doc_name: str) -> str:
        """
        Resolve the PDF path for a document name.

        Tries the ``pdf_dir`` first with exact name, then with ``.pdf``
        appended.  Falls back to empty string if not found.

        Args:
            doc_name: Document name from the registry.

        Returns:
            Absolute path to the PDF, or empty string if not found.
        """
        pdf_dir = self.config.pdf_dir

        # Try exact name (doc_name might already have .pdf)
        candidate = os.path.join(pdf_dir, doc_name)
        if os.path.exists(candidate):
            return candidate

        # Try with .pdf extension
        if not doc_name.endswith(".pdf"):
            candidate = os.path.join(pdf_dir, doc_name + ".pdf")
            if os.path.exists(candidate):
                return candidate

        # Try stripping known suffixes from doc_name
        # e.g. "earthmover_structure.json" → "earthmover.pdf"
        stem = doc_name
        for suffix in ("_structure.json", "_structure", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        candidate = os.path.join(pdf_dir, stem + ".pdf")
        if os.path.exists(candidate):
            return candidate

        logger.warning("_find_pdf_path  PDF not found for doc=%s", doc_name)
        return ""

    # ------------------------------------------------------------------
    # Internal: merge grep hits + context into ContextMerger
    # ------------------------------------------------------------------

    @staticmethod
    def _add_to_merger(
        merger: ContextMerger,
        doc_name: str,
        grep_hits: List[GrepResult],
        context_parts: List[str],
    ) -> None:
        """
        Convert grep hits and fetched context parts into ContextChunks
        and add them to the merger.

        Strategy:
        - Build a lookup from node_id → fetched context text.
        - For each grep hit, if we have fetched body text for that node,
          create a ContextChunk with the full body and the grep relevance score.
        - If we have fetched text for nodes not in grep hits (shouldn't
          happen, but defensive), add them with score 1.0.

        Args:
            merger:        The ContextMerger to add chunks to.
            doc_name:      Document name for tagging.
            grep_hits:     TreeGrep search results.
            context_parts: Fetched context strings from ContextFetcher.
        """
        # Build node_id → context text lookup from fetcher output
        # Fetcher output format: "## Section: Title (Pages X-Y) [node_id: XXXX]\n\nbody"
        node_context: Dict[str, str] = {}
        node_titles: Dict[str, str] = {}
        node_pages: Dict[str, Tuple[int, int]] = {}

        import re

        header_re = re.compile(
            r"^##\s+Section:\s+(.+?)\s+\(Pages\s+(\d+)-(\d+)\)\s+\[node_id:\s*(\S+)\]"
        )

        for part in context_parts:
            match = header_re.match(part)
            if match:
                title = match.group(1)
                start_page = int(match.group(2))
                end_page = int(match.group(3))
                nid = match.group(4).rstrip("]")
                # Body is everything after the header
                body_start = part.index("\n", match.end())
                body = part[body_start:].strip()
                node_context[nid] = body
                node_titles[nid] = title
                node_pages[nid] = (start_page, end_page)
            else:
                # Fallback: use the entire part as body
                node_context[f"_unknown_{len(node_context)}"] = part

        # Build a set of node_ids we've already added
        added_nodes: set = set()

        # Add chunks for grep hits (with their relevance scores)
        for hit in grep_hits:
            if hit.node_id in added_nodes:
                continue  # dedup within this doc
            added_nodes.add(hit.node_id)

            body = node_context.get(hit.node_id, "")
            if not body:
                # No fetched body for this node — use the grep snippet
                body = hit.snippet

            merger.add_chunk(
                ContextChunk(
                    doc_name=doc_name,
                    node_id=hit.node_id,
                    title=node_titles.get(hit.node_id, hit.title),
                    page_range=node_pages.get(hit.node_id, hit.page_range),
                    text=body,
                    relevance_score=hit.relevance_score,
                )
            )

        # Add any fetched nodes not covered by grep hits
        for nid, body in node_context.items():
            if nid not in added_nodes and not nid.startswith("_unknown_"):
                merger.add_chunk(
                    ContextChunk(
                        doc_name=doc_name,
                        node_id=nid,
                        title=node_titles.get(nid, "Untitled"),
                        page_range=node_pages.get(nid, (0, 0)),
                        text=body,
                        relevance_score=1.0,  # default for un-grepped nodes
                    )
                )

    # ------------------------------------------------------------------
    # Internal: citations
    # ------------------------------------------------------------------

    @staticmethod
    def _build_citations(merged: MergedContext) -> List[Dict[str, Any]]:
        """
        Extract citation metadata from merged chunks.

        Returns a list of dicts, each with:
        - ``doc_name``
        - ``section`` (title)
        - ``pages`` (``"start-end"``)
        - ``node_id``
        """
        citations: List[Dict[str, Any]] = []
        seen: set = set()
        for chunk in merged.chunks:
            key = (chunk.doc_name, chunk.node_id)
            if key in seen:
                continue
            seen.add(key)
            start, end = chunk.page_range
            citations.append(
                {
                    "doc_name": chunk.doc_name,
                    "section": chunk.title,
                    "pages": f"{start}-{end}",
                    "node_id": chunk.node_id,
                }
            )
        return citations

    # ------------------------------------------------------------------
    # Convenience: summary
    # ------------------------------------------------------------------

    def corpus_summary(self) -> str:
        """Return a human-readable summary of the current corpus."""
        return self.registry.to_llm_summary()

    def corpus_stats(self) -> Dict[str, Any]:
        """Return corpus statistics."""
        return {
            "num_documents": len(self.registry.doc_names),
            "doc_names": self.registry.doc_names,
            "changelog_entries": len(self.changelog.entries),
            "active_docs": self.changelog.get_active_docs(),
        }
