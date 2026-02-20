"""
MdExtractor: Extract PDF text into structured Markdown files.

Uses PyMuPDF (fitz) to extract text from PDF pages, then maps the extracted
text onto the document's structure JSON (node tree with page ranges) to produce
a Markdown file where:
  - Each node becomes a heading (depth = nesting level)
  - An HTML comment carries metadata: node_id, page range
  - Body text is the NFC-normalized content from those pages

The .md file becomes the single source of truth for all downstream search
and retrieval — no more re-reading the PDF at query time.
"""

import json
import logging
import os
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class MdExtractor:
    """
    Convert a PDF + structure JSON into a structured Markdown document.

    Usage:
        extractor = MdExtractor(pdf_path="doc.pdf", index_path="doc_structure.json")
        md_path = extractor.extract()           # returns path to written .md file
        md_text = extractor.extract_to_string() # returns the markdown as string
    """

    def __init__(
        self,
        pdf_path: str,
        index_path: str,
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            pdf_path:   Path to the source PDF file.
            index_path: Path to the PageIndex structure JSON.
            output_dir: Directory for the output .md file.
                        Defaults to the same directory as index_path.
        """
        self.pdf_path = pdf_path
        self.index_path = index_path
        self.output_dir = output_dir or os.path.dirname(index_path) or "."

        logger.debug(
            "MdExtractor init  pdf=%s  index=%s  output_dir=%s",
            self.pdf_path,
            self.index_path,
            self.output_dir,
        )

        # Load the structure JSON
        with open(self.index_path, "r", encoding="utf-8") as f:
            self.index: Dict[str, Any] = json.load(f)

        self.doc_name: str = self.index.get("doc_name", "unknown")
        self.structure: List[dict] = self.index.get("structure", [])

        logger.debug(
            "MdExtractor loaded index  doc_name=%s  top_level_nodes=%d",
            self.doc_name,
            len(self.structure),
        )

    # ------------------------------------------------------------------
    # PDF text extraction
    # ------------------------------------------------------------------

    def _extract_page_texts(self) -> Dict[int, str]:
        """
        Extract text from every page of the PDF.

        Returns:
            dict mapping 1-based page number -> NFC-normalized text string.
        """
        logger.debug("MdExtractor._extract_page_texts  opening %s", self.pdf_path)
        page_texts: Dict[int, str] = {}

        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)
        logger.debug("MdExtractor  total PDF pages=%d", total_pages)

        for page_num in range(total_pages):
            page = doc[page_num]
            raw_text = page.get_text("text")  # plain-text extraction
            # NFC normalize for Nepali/Devanagari safety
            normalized = unicodedata.normalize("NFC", raw_text)
            # Clean up excessive whitespace while preserving paragraph breaks
            cleaned = self._clean_text(normalized)
            page_texts[page_num + 1] = cleaned  # 1-based key
            logger.debug(
                "MdExtractor  page %d/%d  chars=%d",
                page_num + 1,
                total_pages,
                len(cleaned),
            )

        doc.close()
        logger.info(
            "MdExtractor._extract_page_texts  done  pages=%d  total_chars=%d",
            total_pages,
            sum(len(t) for t in page_texts.values()),
        )
        return page_texts

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Collapse runs of blank lines to at most two newlines (paragraph break)
        and strip trailing whitespace from each line.
        """
        lines = text.split("\n")
        cleaned_lines = [line.rstrip() for line in lines]
        # Collapse 3+ consecutive blank lines into a single blank line
        result: List[str] = []
        blank_count = 0
        for line in cleaned_lines:
            if line == "":
                blank_count += 1
                if blank_count <= 1:
                    result.append(line)
            else:
                blank_count = 0
                result.append(line)
        return "\n".join(result)

    # ------------------------------------------------------------------
    # Text for a page range
    # ------------------------------------------------------------------

    def _get_text_for_range(
        self,
        page_texts: Dict[int, str],
        start_page: int,
        end_page: int,
    ) -> str:
        """
        Concatenate text for pages [start_page, end_page] (inclusive, 1-based).
        """
        parts: List[str] = []
        for p in range(start_page, end_page + 1):
            text = page_texts.get(p, "")
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Recursive tree -> Markdown
    # ------------------------------------------------------------------

    def _node_to_markdown(
        self,
        node: dict,
        page_texts: Dict[int, str],
        depth: int,
        used_pages: Dict[int, bool],
    ) -> str:
        """
        Recursively convert a structure node (and its children) into Markdown.

        Args:
            node:       A node from the structure tree.
            page_texts: Pre-extracted page text dict.
            depth:      Current heading depth (1 = top-level).
            used_pages: Tracks which pages have been claimed by child nodes
                        to avoid duplicating text between parent and children.

        Returns:
            Markdown string for this node and its descendants.
        """
        node_id = node.get("node_id", "????")
        title = node.get("title", "Untitled")
        start_page = node.get("start_index", 0)
        end_page = node.get("end_index", 0)
        children = node.get("nodes") or []

        # Clamp heading depth to h1..h6
        heading_level = min(depth, 6)
        heading_prefix = "#" * heading_level

        logger.debug(
            "MdExtractor._node_to_markdown  node_id=%s  title=%r  "
            "pages=%d-%d  depth=%d  children=%d",
            node_id,
            title,
            start_page,
            end_page,
            depth,
            len(children),
        )

        lines: List[str] = []

        # HTML comment with metadata
        lines.append(f"<!-- node_id: {node_id} | pages: {start_page}-{end_page} -->")
        # Heading
        lines.append(f"{heading_prefix} {title}")
        lines.append("")  # blank line after heading

        if children:
            # When a node has children, figure out which pages belong
            # exclusively to the parent (before the first child starts)
            # vs which pages are covered by children.
            child_page_ranges = []
            for child in children:
                cs = child.get("start_index", 0)
                ce = child.get("end_index", 0)
                child_page_ranges.append((cs, ce))

            # Pages belonging to this parent but not any child:
            # the "intro" text before children begin
            first_child_start = (
                min(cs for cs, _ in child_page_ranges)
                if child_page_ranges
                else start_page
            )
            if start_page < first_child_start:
                intro_text = self._get_text_for_range(
                    page_texts, start_page, first_child_start - 1
                )
                if intro_text.strip():
                    lines.append(intro_text)
                    lines.append("")
            elif start_page == first_child_start:
                # Parent and first child share the same starting page.
                # We'll let the children handle the text to avoid duplication.
                pass

            # Mark child pages as used
            for cs, ce in child_page_ranges:
                for p in range(cs, ce + 1):
                    used_pages[p] = True

            # Recurse into children
            for child in children:
                child_md = self._node_to_markdown(
                    child, page_texts, depth + 1, used_pages
                )
                lines.append(child_md)
        else:
            # Leaf node — emit the page text
            text = self._get_text_for_range(page_texts, start_page, end_page)
            if text.strip():
                lines.append(text)
                lines.append("")
            # Mark these pages as used
            for p in range(start_page, end_page + 1):
                used_pages[p] = True

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_to_string(self) -> str:
        """
        Extract PDF text and return the full structured Markdown as a string.
        """
        logger.info("MdExtractor.extract_to_string  start  doc=%s", self.doc_name)

        page_texts = self._extract_page_texts()
        used_pages: Dict[int, bool] = {}

        parts: List[str] = []

        # Document-level header
        doc_description = self.index.get("doc_description", "")
        parts.append(f"<!-- doc_name: {self.doc_name} -->")
        if doc_description:
            parts.append(f"<!-- doc_description: {doc_description} -->")
        parts.append("")

        # Walk the structure tree
        for node in self.structure:
            node_md = self._node_to_markdown(
                node, page_texts, depth=1, used_pages=used_pages
            )
            parts.append(node_md)

        # Catch any orphan pages not covered by the structure tree
        all_pages = set(page_texts.keys())
        covered_pages = {p for p, used in used_pages.items() if used}
        orphan_pages = sorted(all_pages - covered_pages)
        if orphan_pages:
            logger.warning(
                "MdExtractor  %d orphan pages not in structure: %s",
                len(orphan_pages),
                orphan_pages,
            )
            parts.append("")
            parts.append("<!-- orphan_pages: uncovered by structure tree -->")
            parts.append("# Appendix (Uncovered Pages)")
            parts.append("")
            for p in orphan_pages:
                text = page_texts.get(p, "")
                if text.strip():
                    parts.append(f"<!-- page: {p} -->")
                    parts.append(text)
                    parts.append("")

        md_content = "\n".join(parts)
        logger.info(
            "MdExtractor.extract_to_string  done  total_chars=%d  orphan_pages=%d",
            len(md_content),
            len(orphan_pages),
        )
        return md_content

    def extract(self, output_filename: Optional[str] = None) -> str:
        """
        Extract PDF text and write the structured Markdown to a file.

        Args:
            output_filename: Override the output filename.
                             Defaults to '<doc_stem>.md'.

        Returns:
            The absolute path to the written .md file.
        """
        if output_filename is None:
            stem = Path(self.doc_name).stem
            output_filename = f"{stem}.md"

        output_path = os.path.join(self.output_dir, output_filename)
        logger.info(
            "MdExtractor.extract  writing to %s",
            output_path,
        )

        md_content = self.extract_to_string()

        os.makedirs(self.output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        file_size = os.path.getsize(output_path)
        logger.info(
            "MdExtractor.extract  done  path=%s  size=%d bytes",
            output_path,
            file_size,
        )
        return output_path


def extract_markdown(
    pdf_path: str,
    index_path: str,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> str:
    """
    Convenience function: extract PDF to Markdown in one call.

    Returns:
        Path to the written .md file.
    """
    extractor = MdExtractor(
        pdf_path=pdf_path,
        index_path=index_path,
        output_dir=output_dir,
    )
    return extractor.extract(output_filename=output_filename)
