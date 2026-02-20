"""
Context Fetcher — PDF and Markdown context extraction.

This module provides the ``ContextFetcher`` class which extracts text from
either PDFs (v1 path) or cached Markdown files (v2 path) based on page
ranges / node_ids defined in the hierarchical document index.

v2 additions:
  - ``fetch_from_md()`` — reads context directly from a structured ``.md``
    file produced by :class:`ldrs.md_extractor.MdExtractor`, avoiding
    costly PDF re-reads at query time.
  - Logging throughout (``logging.DEBUG``)
  - NFC normalization of returned text

The original ``fetch()`` method (PDF-based) is preserved for backward
compatibility and as a fallback.

Example (v2 preferred path)::

    fetcher = ContextFetcher(
        index_path="results/earthmover_structure.json",
        pdf_path="earthmover.pdf"          # still needed for fallback
    )
    # v2: read from cached .md
    context_list = fetcher.fetch_from_md(
        node_ids=["0003", "0007"],
        md_path="results/earthmover.md"
    )

    # v1 fallback: read from PDF
    context_list = fetcher.fetch(["0003", "0007"])
"""

import json
import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional

import PyPDF2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown section parser (shared with tree_grep)
# ---------------------------------------------------------------------------


def _parse_md_sections(md_text: str) -> Dict[str, str]:
    """
    Parse a structured Markdown file into ``{node_id: body_text}``.

    The ``.md`` format uses HTML comments as section markers::

        <!-- node_id: 0003 | pages: 5-7 -->

    Everything between one marker and the next (or EOF) is that node's body.

    Args:
        md_text: The full Markdown text.

    Returns:
        Dict mapping node_id strings to their body text (NFC-normalized).
    """
    sections: Dict[str, str] = {}
    marker_re = re.compile(r"<!--\s*node_id:\s*(\S+)\s*\|")

    parts = marker_re.split(md_text)
    idx = 1
    while idx < len(parts) - 1:
        node_id = parts[idx].strip()
        body = unicodedata.normalize("NFC", parts[idx + 1].strip())
        sections[node_id] = body
        idx += 2

    logger.debug("_parse_md_sections  parsed %d sections", len(sections))
    return sections


# ---------------------------------------------------------------------------
# ContextFetcher class
# ---------------------------------------------------------------------------


class ContextFetcher:
    """
    Extracts text context for document nodes, from either PDF or Markdown.

    This class provides methods to:

    - ``fetch(node_ids)`` — extract from PDF via PyPDF2 (v1 path)
    - ``fetch_from_md(node_ids, md_path)`` — extract from cached ``.md`` (v2 path)
    - ``get_node_info(node_ids)`` — metadata about retrieved nodes

    The class works with 1-indexed page numbers (user-friendly) but
    internally converts to 0-indexed for PyPDF2.

    Attributes:
        pdf_path:  Path to the source PDF file.
        index:     The loaded JSON index structure.
        node_map:  Flattened mapping of ``node_id -> node data``.

    Args:
        index_path: Path to the JSON index file.
        pdf_path:   Path to the PDF file.

    Raises:
        FileNotFoundError: If *index_path* does not exist.
    """

    def __init__(self, index_path: str, pdf_path: str):
        """
        Initialize the context fetcher.

        Loads the index file, then builds a flattened node map for lookups.
        The PDF is **not** opened at init time — only when ``fetch()`` is called.
        """
        logger.debug(
            "ContextFetcher init  index_path=%s  pdf_path=%s", index_path, pdf_path
        )
        self.pdf_path = pdf_path
        self.index_path = index_path

        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)

        self.node_map: Dict[str, dict] = self._build_node_map(
            self.index.get("structure", [])
        )
        logger.debug("ContextFetcher  node_map has %d nodes", len(self.node_map))

    # ------------------------------------------------------------------
    # Node map builder
    # ------------------------------------------------------------------

    def _build_node_map(
        self,
        nodes: List[dict],
        result: Optional[Dict[str, dict]] = None,  # noqa: type-ignore
    ) -> Dict[str, dict]:
        """
        Recursively flatten the tree into a ``node_id -> node`` mapping.

        Args:
            nodes:  List of nodes to process.
            result: Accumulator dictionary (used for recursion).

        Returns:
            Dictionary mapping node_id to node data (title, pages, summary).
        """
        if result is None:
            result = {}

        for node in nodes:
            nid = node.get("node_id")
            if nid:
                result[nid] = {
                    "title": node.get("title", ""),
                    "start_index": node.get("start_index"),
                    "end_index": node.get("end_index"),
                    "summary": node.get("summary", ""),
                }
            if node.get("nodes"):
                self._build_node_map(node["nodes"], result)

        return result

    # ------------------------------------------------------------------
    # v1 path: PDF-based context extraction
    # ------------------------------------------------------------------

    def _get_page_text(self, start_page: int, end_page: int) -> str:
        """
        Extract text from a range of PDF pages using PyPDF2.

        Uses 1-indexed page numbers for consistency with the index.
        Includes page markers in the output for reference.

        Args:
            start_page: Starting page number (1-indexed).
            end_page:   Ending page number (1-indexed, inclusive).

        Returns:
            String containing extracted text with ``--- Page N ---`` markers.
        """
        logger.debug(
            "_get_page_text  pages %d-%d  pdf=%s", start_page, end_page, self.pdf_path
        )
        pdf_reader = PyPDF2.PdfReader(self.pdf_path)
        text = ""

        for page_num in range(start_page - 1, min(end_page, len(pdf_reader.pages))):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text() or ""
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"

        return text

    def fetch(self, node_ids: List[str]) -> List[str]:
        """
        Fetch context from selected nodes by reading the PDF.

        **v1 path** — opens the PDF and extracts text for each node's page
        range.  For v2, prefer :meth:`fetch_from_md` which reads from the
        cached Markdown file instead.

        Args:
            node_ids: List of node IDs to fetch context for.

        Returns:
            List of context strings (one per node, with section headers).
            Returns empty list if no valid node IDs provided.
        """
        logger.info("ContextFetcher.fetch (PDF)  node_ids=%s", node_ids)
        if not node_ids:
            return []

        context_parts: List[str] = []

        for node_id in node_ids:
            if node_id not in self.node_map:
                logger.warning("ContextFetcher.fetch  unknown node_id=%s", node_id)
                continue

            node = self.node_map[node_id]
            start = node.get("start_index")
            end = node.get("end_index")

            if start is None or end is None:
                logger.warning(
                    "ContextFetcher.fetch  node %s has no page range", node_id
                )
                continue

            title = node.get("title", "Untitled")
            text = self._get_page_text(start, end)

            context_parts.append(f"## Section: {title} (Pages {start}-{end})\n{text}")

        logger.info(
            "ContextFetcher.fetch  returned %d context parts", len(context_parts)
        )
        return context_parts

    # ------------------------------------------------------------------
    # v2 path: Markdown-based context extraction
    # ------------------------------------------------------------------

    def fetch_from_md(
        self,
        node_ids: List[str],
        md_path: str,
        max_chars_per_node: int = 0,
    ) -> List[str]:
        """
        Fetch context from selected nodes by reading the cached Markdown file.

        **v2 preferred path** — reads from the structured ``.md`` file
        produced by :class:`ldrs.md_extractor.MdExtractor`.  Much faster
        than PDF extraction and the text is already NFC-normalized.

        Each returned string includes a header with the node title and
        page range, followed by the Markdown body text.

        Args:
            node_ids:           List of node IDs to fetch context for.
            md_path:            Path to the structured ``.md`` file.
            max_chars_per_node: If > 0, truncate each node's body to this
                                many characters (with a ``[truncated]`` marker).

        Returns:
            List of context strings (one per node found in the .md file).
            Returns empty list if no valid node IDs or .md file is missing.

        Example::

            parts = fetcher.fetch_from_md(
                node_ids=["0003", "0007"],
                md_path="results/earthmover.md",
                max_chars_per_node=4000,
            )
        """
        logger.info(
            "ContextFetcher.fetch_from_md  node_ids=%s  md_path=%s  max_chars=%d",
            node_ids,
            md_path,
            max_chars_per_node,
        )

        if not node_ids:
            return []

        if not os.path.exists(md_path):
            logger.warning(
                "ContextFetcher.fetch_from_md  md file not found: %s  "
                "falling back to PDF fetch",
                md_path,
            )
            return self.fetch(node_ids)

        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        sections = _parse_md_sections(md_text)
        logger.debug(
            "ContextFetcher.fetch_from_md  parsed %d sections from .md",
            len(sections),
        )

        context_parts: List[str] = []

        for node_id in node_ids:
            # Get metadata from the node map
            node_meta = self.node_map.get(node_id)
            title = node_meta.get("title", "Untitled") if node_meta else "Untitled"
            start = node_meta.get("start_index", "?") if node_meta else "?"
            end = node_meta.get("end_index", "?") if node_meta else "?"

            body = sections.get(node_id, "")
            if not body:
                logger.debug(
                    "ContextFetcher.fetch_from_md  node %s not in .md sections",
                    node_id,
                )
                continue

            # Optional truncation
            if max_chars_per_node > 0 and len(body) > max_chars_per_node:
                body = body[:max_chars_per_node] + "\n\n[... truncated]"
                logger.debug(
                    "ContextFetcher.fetch_from_md  truncated node %s to %d chars",
                    node_id,
                    max_chars_per_node,
                )

            header = f"## Section: {title} (Pages {start}-{end}) [node_id: {node_id}]"
            context_parts.append(f"{header}\n\n{body}")

        logger.info(
            "ContextFetcher.fetch_from_md  returned %d context parts",
            len(context_parts),
        )
        return context_parts

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_node_info(self, node_ids: List[str]) -> List[dict]:
        """
        Get metadata for retrieved nodes.

        Useful for debugging and for including in response metadata.

        Args:
            node_ids: List of node IDs to get info for.

        Returns:
            List of dicts with ``node_id``, ``title``, and ``pages`` keys.
        """
        return [
            {
                "node_id": nid,
                "title": self.node_map[nid]["title"],
                "pages": f"{self.node_map[nid]['start_index']}-{self.node_map[nid]['end_index']}",
            }
            for nid in node_ids
            if nid in self.node_map
        ]
