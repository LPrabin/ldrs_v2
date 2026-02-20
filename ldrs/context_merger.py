"""
ContextMerger: Cross-document context ranking, deduplication, and budgeting.

In a multi-document retrieval scenario, multiple documents produce context
chunks from ``TreeGrep`` hits and ``ContextFetcher`` results.  The merger:

1. **Collects** context chunks from all documents, each tagged with its
   source document, node_id, relevance score, and text.
2. **Deduplicates** chunks that overlap (same node_id from the same doc).
3. **Ranks** chunks by relevance score (from TreeGrep) so the most
   important content appears first.
4. **Budgets** total text to a configurable character / token limit,
   truncating or dropping low-priority chunks as needed.
5. **Formats** the final merged context string for the LLM generator.

This module is entirely synchronous and does not call any LLM.

Usage::

    merger = ContextMerger(max_total_chars=12000)

    # Add chunks from multiple documents
    merger.add_chunks("earthmover.pdf", chunks_from_earthmover)
    merger.add_chunks("report.pdf", chunks_from_report)

    # Get the merged, ranked, budgeted context
    merged = merger.merge()
    print(merged.formatted_context)
    print(f"Used {merged.total_chars} chars from {merged.num_chunks} chunks")
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOTAL_CHARS = 15_000
"""Default character budget for the merged context."""

DEFAULT_MAX_CHUNKS = 30
"""Default maximum number of context chunks to keep."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ContextChunk:
    """
    A single piece of retrieved context from one document.

    Attributes:
        doc_name:        Source document name (e.g. ``"earthmover.pdf"``).
        node_id:         The node_id this chunk belongs to.
        title:           Section title.
        page_range:      ``(start_page, end_page)`` 1-indexed.
        text:            The actual context text (body content).
        relevance_score: Numeric relevance from TreeGrep (higher = better).
        matched_query:   Which sub-query triggered this hit (optional).
    """

    doc_name: str
    node_id: str
    title: str
    page_range: Tuple[int, int]
    text: str
    relevance_score: float = 1.0
    matched_query: str = ""

    @property
    def char_count(self) -> int:
        """Character count of the text body."""
        return len(self.text)

    @property
    def dedup_key(self) -> Tuple[str, str]:
        """Key for deduplication: ``(doc_name, node_id)``."""
        return (self.doc_name, self.node_id)


@dataclass
class MergedContext:
    """
    The result of merging and budgeting context chunks.

    Attributes:
        chunks:            Final list of chunks (sorted, deduped, budgeted).
        formatted_context: Ready-to-use string for the LLM prompt.
        total_chars:       Total character count of formatted_context.
        num_chunks:        Number of chunks included.
        num_docs:          Number of distinct documents represented.
        dropped_count:     Number of chunks dropped due to budget.
    """

    chunks: List[ContextChunk] = field(default_factory=list)
    formatted_context: str = ""
    total_chars: int = 0
    num_chunks: int = 0
    num_docs: int = 0
    dropped_count: int = 0


# ---------------------------------------------------------------------------
# ContextMerger class
# ---------------------------------------------------------------------------


class ContextMerger:
    """
    Merge, rank, deduplicate, and budget context from multiple documents.

    The merger accumulates :class:`ContextChunk` objects via :meth:`add_chunk`
    or :meth:`add_chunks`, then produces a :class:`MergedContext` via
    :meth:`merge`.

    Args:
        max_total_chars: Maximum total characters in the merged output.
        max_chunks:      Maximum number of chunks to keep.

    Example::

        merger = ContextMerger(max_total_chars=12000)
        merger.add_chunk(ContextChunk(
            doc_name="report.pdf",
            node_id="0003",
            title="Revenue",
            page_range=(5, 7),
            text="Revenue was $1.2B...",
            relevance_score=3.0,
        ))
        result = merger.merge()
    """

    def __init__(
        self,
        max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
    ):
        self.max_total_chars = max_total_chars
        self.max_chunks = max_chunks
        self._chunks: List[ContextChunk] = []

        logger.debug(
            "ContextMerger init  max_chars=%d  max_chunks=%d",
            self.max_total_chars,
            self.max_chunks,
        )

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def add_chunk(self, chunk: ContextChunk) -> None:
        """Add a single context chunk to the merge pool."""
        self._chunks.append(chunk)
        logger.debug(
            "ContextMerger.add_chunk  doc=%s  node=%s  score=%.1f  chars=%d",
            chunk.doc_name,
            chunk.node_id,
            chunk.relevance_score,
            chunk.char_count,
        )

    def add_chunks(self, chunks: List[ContextChunk]) -> None:
        """Add multiple context chunks at once."""
        for chunk in chunks:
            self.add_chunk(chunk)

    def add_from_fetcher_output(
        self,
        doc_name: str,
        context_parts: List[str],
        node_ids: List[str],
        relevance_scores: Optional[List[float]] = None,
    ) -> None:
        """
        Convenience method: convert raw ``ContextFetcher.fetch*()`` output
        into :class:`ContextChunk` objects and add them.

        Args:
            doc_name:         Source document name.
            context_parts:    List of context strings from the fetcher.
            node_ids:         Corresponding node_ids (same order as context_parts).
            relevance_scores: Optional relevance scores.  Defaults to 1.0 for
                              each chunk if not provided.
        """
        if relevance_scores is None:
            relevance_scores = [1.0] * len(context_parts)

        for i, (text, nid) in enumerate(zip(context_parts, node_ids)):
            # Try to extract title and page range from the header line
            title, page_range = self._parse_fetcher_header(text)
            score = relevance_scores[i] if i < len(relevance_scores) else 1.0

            self.add_chunk(
                ContextChunk(
                    doc_name=doc_name,
                    node_id=nid,
                    title=title,
                    page_range=page_range,
                    text=text,
                    relevance_score=score,
                )
            )

    @staticmethod
    def _parse_fetcher_header(text: str) -> Tuple[str, Tuple[int, int]]:
        """
        Extract title and page range from a ``ContextFetcher`` output header.

        Expected format::

            ## Section: Some Title (Pages 5-7)
            ...body...

        Returns:
            ``(title, (start_page, end_page))``  or  ``("Untitled", (0, 0))``
        """
        match = re.match(
            r"^##\s+Section:\s+(.+?)\s+\(Pages\s+(\d+)-(\d+)\)",
            text,
        )
        if match:
            return match.group(1), (int(match.group(2)), int(match.group(3)))
        return ("Untitled", (0, 0))

    # ------------------------------------------------------------------
    # Merge pipeline
    # ------------------------------------------------------------------

    def merge(self) -> MergedContext:
        """
        Run the full merge pipeline: deduplicate → rank → budget → format.

        Returns:
            :class:`MergedContext` with the final ranked context and metadata.
        """
        logger.info(
            "ContextMerger.merge  starting with %d raw chunks", len(self._chunks)
        )

        if not self._chunks:
            logger.info("ContextMerger.merge  no chunks to merge")
            return MergedContext()

        # Step 1: Deduplicate
        deduped = self._deduplicate(self._chunks)
        logger.debug("ContextMerger.merge  after dedup: %d chunks", len(deduped))

        # Step 2: Sort by relevance (descending), then by document order
        ranked = sorted(
            deduped,
            key=lambda c: (-c.relevance_score, c.doc_name, c.node_id),
        )

        # Step 3: Apply chunk count cap
        if len(ranked) > self.max_chunks:
            logger.debug(
                "ContextMerger.merge  capping chunks %d -> %d",
                len(ranked),
                self.max_chunks,
            )
            dropped_by_cap = len(ranked) - self.max_chunks
            ranked = ranked[: self.max_chunks]
        else:
            dropped_by_cap = 0

        # Step 4: Apply character budget
        budgeted, dropped_by_budget = self._apply_budget(ranked)
        total_dropped = dropped_by_cap + dropped_by_budget

        # Step 5: Format
        formatted = self._format(budgeted)

        # Compute stats
        doc_names = set(c.doc_name for c in budgeted)
        result = MergedContext(
            chunks=budgeted,
            formatted_context=formatted,
            total_chars=len(formatted),
            num_chunks=len(budgeted),
            num_docs=len(doc_names),
            dropped_count=total_dropped,
        )

        logger.info(
            "ContextMerger.merge  done  chunks=%d  docs=%d  chars=%d  dropped=%d",
            result.num_chunks,
            result.num_docs,
            result.total_chars,
            result.dropped_count,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(chunks: List[ContextChunk]) -> List[ContextChunk]:
        """
        Remove duplicate chunks (same doc + node_id).

        When duplicates exist, keep the one with the highest relevance_score.
        """
        best: Dict[Tuple[str, str], ContextChunk] = {}
        for chunk in chunks:
            key = chunk.dedup_key
            if key not in best or chunk.relevance_score > best[key].relevance_score:
                best[key] = chunk
        return list(best.values())

    # ------------------------------------------------------------------
    # Internal: Budget enforcement
    # ------------------------------------------------------------------

    def _apply_budget(
        self,
        chunks: List[ContextChunk],
    ) -> Tuple[List[ContextChunk], int]:
        """
        Keep chunks until the character budget is exhausted.

        Chunks are already sorted by relevance.  We greedily include
        chunks from the front.  If a chunk would exceed the budget, we
        try to include a truncated version (keeping at least 200 chars).
        If even 200 chars won't fit, we drop it.

        Args:
            chunks: Pre-sorted list of chunks.

        Returns:
            Tuple of ``(kept_chunks, dropped_count)``.
        """
        kept: List[ContextChunk] = []
        chars_used = 0
        dropped = 0
        min_useful = 200  # minimum chars for a chunk to be worth including

        for chunk in chunks:
            remaining = self.max_total_chars - chars_used
            if remaining < min_useful:
                dropped += 1
                continue

            if chunk.char_count <= remaining:
                # Fits entirely
                kept.append(chunk)
                chars_used += chunk.char_count
            else:
                # Truncate to fit
                truncated_text = chunk.text[: remaining - 30] + "\n\n[... truncated]"
                kept.append(
                    ContextChunk(
                        doc_name=chunk.doc_name,
                        node_id=chunk.node_id,
                        title=chunk.title,
                        page_range=chunk.page_range,
                        text=truncated_text,
                        relevance_score=chunk.relevance_score,
                        matched_query=chunk.matched_query,
                    )
                )
                chars_used += len(truncated_text)
                # Remaining budget is now spent
                # Continue to count remaining as dropped
                dropped += 0  # this one was included (truncated)

        logger.debug(
            "ContextMerger._apply_budget  kept=%d  dropped=%d  chars=%d/%d",
            len(kept),
            dropped,
            chars_used,
            self.max_total_chars,
        )
        return kept, dropped

    # ------------------------------------------------------------------
    # Internal: Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format(chunks: List[ContextChunk]) -> str:
        """
        Format the final list of chunks into a single context string.

        Output format::

            === DOCUMENT: earthmover.pdf ===

            ## Section: Cost Estimation (Pages 5-7) [node_id: 0003]

            <body text>

            ---

            ## Section: Budget (Pages 8-9) [node_id: 0005]

            <body text>

            === DOCUMENT: report.pdf ===
            ...

        Chunks are grouped by document to maintain reading coherence.
        """
        if not chunks:
            return ""

        # Group by doc_name (preserving chunk order within each doc)
        doc_groups: Dict[str, List[ContextChunk]] = {}
        doc_order: List[str] = []  # preserve first-seen order
        for chunk in chunks:
            if chunk.doc_name not in doc_groups:
                doc_groups[chunk.doc_name] = []
                doc_order.append(chunk.doc_name)
            doc_groups[chunk.doc_name].append(chunk)

        parts: List[str] = []
        for doc_name in doc_order:
            parts.append(f"=== DOCUMENT: {doc_name} ===\n")
            for chunk in doc_groups[doc_name]:
                start, end = chunk.page_range
                header = (
                    f"## {chunk.title} (Pages {start}-{end}) [node_id: {chunk.node_id}]"
                )
                parts.append(f"{header}\n\n{chunk.text}")
            parts.append("")  # blank line between docs

        return "\n---\n\n".join(parts).strip()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated chunks."""
        self._chunks.clear()
        logger.debug("ContextMerger.reset  cleared all chunks")
