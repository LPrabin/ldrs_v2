"""
PdfExtractor: Extract entire PDF text into Markdown with heading detection.

Uses PyMuPDF (fitz) to extract full document text from a PDF, analysing
font sizes to detect headings and produce a well-structured Markdown file.

The output Markdown:
  - Contains the **entire document text** (not summaries).
  - Uses ``#``, ``##``, ``###`` etc. for detected headings based on font size.
  - Embeds page markers (``<!-- page: N -->``) so that downstream processing
    can map markdown line numbers back to PDF page numbers.
  - Applies UTF-8 NFC normalization at every text boundary.

This module replaces :class:`MdExtractor`, which required a pre-existing
structure JSON.  ``PdfExtractor`` needs only the PDF file itself.

The typical pipeline flow is::

    PDF  -->  PdfExtractor  -->  .md file
         -->  page_index_md.md_to_tree()  -->  structure JSON
         -->  pipeline uses both .md + structure JSON

Usage::

    extractor = PdfExtractor(pdf_path="doc.pdf", output_dir="results/")
    md_path = extractor.extract()            # writes .md, returns path
    md_text = extractor.extract_to_string()  # returns markdown as string
"""

import logging
import os
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import pytesseract
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum font-size ratio above the body font to qualify as a heading.
# E.g., 1.15 means font must be at least 15% larger than the dominant body font.
HEADING_MIN_RATIO = 1.15

# Maximum heading levels to detect (h1..h4).
MAX_HEADING_LEVELS = 4

# Spans shorter than this are ignored for font-size analysis (e.g. footnote
# numbers, superscripts).
MIN_SPAN_LENGTH = 3

# Lines that are entirely whitespace or very short are skipped as heading
# candidates.
MIN_HEADING_CHARS = 2

# Maximum length for a heading line.  Lines longer than this are likely
# paragraphs set in a larger font, not headings.
MAX_HEADING_CHARS = 200


class PdfExtractor:
    """
    Convert a PDF into a Markdown document with detected headings.

    Analyses font sizes across the document to determine which text spans
    are headings vs body text, then emits Markdown with ``#`` / ``##`` /
    ``###`` headings and ``<!-- page: N -->`` markers.

    Args:
        pdf_path:   Path to the source PDF file.
        output_dir: Directory for the output .md file.
                    Defaults to the same directory as the PDF.

    Usage::

        extractor = PdfExtractor("report.pdf", output_dir="out/")
        md_path = extractor.extract()
    """

    def __init__(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        use_ocr: bool = False,
    ):
        self.pdf_path = pdf_path
        self.output_dir = output_dir or os.path.dirname(pdf_path) or "."
        self.doc_name = os.path.basename(pdf_path)
        self.use_ocr = use_ocr

        logger.debug(
            "PdfExtractor init  pdf=%s  output_dir=%s",
            self.pdf_path,
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Font analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _analyse_fonts(doc: fitz.Document) -> Tuple[float, List[float]]:
        """
        Analyse font sizes across the entire document.

        Returns:
            (body_font_size, heading_thresholds)

            ``body_font_size`` is the most common font size (the dominant
            body text size).

            ``heading_thresholds`` is a sorted list (descending) of distinct
            font sizes that are significantly larger than the body font.
            These become the heading levels (largest = h1, next = h2, ...).
        """
        # Collect font sizes weighted by character count
        size_char_count: Counter = Counter()

        for page in doc:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)[
                "blocks"
            ]
            for block in blocks:
                if block.get("type") != 0:  # skip image blocks
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if len(text) < MIN_SPAN_LENGTH:
                            continue
                        font_size = round(span.get("size", 0), 1)
                        if font_size > 0:
                            size_char_count[font_size] += len(text)

        if not size_char_count:
            return 12.0, []  # fallback

        # Body font = the size with the most total characters
        body_size = size_char_count.most_common(1)[0][0]

        # Heading sizes = anything significantly larger than body
        threshold = body_size * HEADING_MIN_RATIO
        heading_sizes = sorted(
            [s for s in size_char_count if s >= threshold],
            reverse=True,
        )

        # Limit to MAX_HEADING_LEVELS distinct sizes
        heading_sizes = heading_sizes[:MAX_HEADING_LEVELS]

        logger.debug(
            "PdfExtractor._analyse_fonts  body_size=%.1f  heading_sizes=%s",
            body_size,
            heading_sizes,
        )
        return body_size, heading_sizes

    @staticmethod
    def _font_size_to_heading_level(
        font_size: float,
        heading_sizes: List[float],
    ) -> int:
        """
        Map a font size to a heading level (1-based), or 0 if not a heading.
        """
        for level, size in enumerate(heading_sizes, start=1):
            if abs(font_size - size) < 0.5:  # tolerance for rounding
                return level
        return 0

    # ------------------------------------------------------------------
    # Text extraction with font metadata
    # ------------------------------------------------------------------

    def _extract_page_with_fonts(
        self,
        page: fitz.Page,
        heading_sizes: List[float],
        body_size: float,
    ) -> List[Dict[str, Any]]:
        """
        Extract text from a single page, annotating each logical line
        with its dominant font size and detected heading level.

        Returns a list of dicts:
            [{"text": str, "font_size": float, "heading_level": int, "is_bold": bool}, ...]
        """
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        lines_out: List[Dict[str, Any]] = []

        for block in blocks:
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                # Build the full line text and determine dominant font size
                line_text_parts: List[str] = []
                size_weights: Counter = Counter()
                bold_chars = 0
                total_chars = 0

                for span in spans:
                    text = span.get("text", "")
                    line_text_parts.append(text)
                    stripped = text.strip()
                    char_count = len(stripped)
                    if char_count > 0:
                        font_size = round(span.get("size", 0), 1)
                        size_weights[font_size] += char_count
                        total_chars += char_count
                        # Check for bold via font name heuristic
                        font_name = span.get("font", "").lower()
                        if "bold" in font_name or "heavy" in font_name:
                            bold_chars += char_count

                full_text = "".join(line_text_parts).rstrip()
                if not full_text.strip():
                    lines_out.append(
                        {
                            "text": "",
                            "font_size": body_size,
                            "heading_level": 0,
                            "is_bold": False,
                        }
                    )
                    continue

                dominant_size = (
                    size_weights.most_common(1)[0][0] if size_weights else body_size
                )
                is_bold = bold_chars > total_chars * 0.5 if total_chars > 0 else False

                heading_level = self._font_size_to_heading_level(
                    dominant_size, heading_sizes
                )

                # Sanity checks: headings shouldn't be too long or too short
                text_len = len(full_text.strip())
                if heading_level > 0:
                    if text_len < MIN_HEADING_CHARS or text_len > MAX_HEADING_CHARS:
                        heading_level = 0

                lines_out.append(
                    {
                        "text": full_text,
                        "font_size": dominant_size,
                        "heading_level": heading_level,
                        "is_bold": is_bold,
                    }
                )

        return lines_out

    # ------------------------------------------------------------------
    # Markdown generation
    # ------------------------------------------------------------------

    def _lines_to_markdown(
        self,
        page_lines: Dict[int, List[Dict[str, Any]]],
    ) -> str:
        """
        Convert extracted page lines (with heading annotations) into a
        Markdown string with page markers.

        Args:
            page_lines: dict mapping 1-based page number to list of
                        annotated line dicts from ``_extract_page_with_fonts``.

        Returns:
            Complete Markdown string.
        """
        parts: List[str] = []

        # Document header
        parts.append(f"<!-- doc_name: {self.doc_name} -->")
        parts.append("")

        for page_num in sorted(page_lines.keys()):
            lines = page_lines[page_num]
            parts.append(f"<!-- page: {page_num} -->")

            prev_was_blank = False
            for line_info in lines:
                text = line_info["text"]
                heading_level = line_info["heading_level"]

                # NFC normalize
                text = unicodedata.normalize("NFC", text)

                if not text.strip():
                    if not prev_was_blank:
                        parts.append("")
                        prev_was_blank = True
                    continue

                prev_was_blank = False

                if heading_level > 0:
                    prefix = "#" * heading_level
                    # Clean heading text — strip leading/trailing whitespace,
                    # remove any existing markdown heading markers
                    clean_title = text.strip().lstrip("#").strip()
                    parts.append("")  # blank line before heading
                    parts.append(f"{prefix} {clean_title}")
                    parts.append("")  # blank line after heading
                else:
                    parts.append(text)

            parts.append("")  # blank line after page

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Clean text helper
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Collapse runs of blank lines to at most one blank line and strip
        trailing whitespace from each line.
        """
        lines = text.split("\n")
        cleaned: List[str] = []
        blank_count = 0
        for line in lines:
            stripped = line.rstrip()
            if stripped == "":
                blank_count += 1
                if blank_count <= 1:
                    cleaned.append("")
            else:
                blank_count = 0
                cleaned.append(stripped)
        return "\n".join(cleaned)

    # ------------------------------------------------------------------
    # OCR Fallback for Nepali documents
    # ------------------------------------------------------------------

    @staticmethod
    def _is_nepali_unicode(text: str) -> bool:
        """
        Check if the string contains a significant amount of Nepali Unicode characters.

        Standard extraction (fitz/PyMuPDF) often succeeds in extracting some
        metadata or small portions in Unicode even if the main body is in a
        legacy font (like Preeti). We check the ratio of Nepali characters
        to ensure the extraction was truly successful for the content.
        """
        # We only check a sample if the text is very large
        sample = text[:10000]
        clean_sample = re.sub(r"\s+", "", sample)
        if not clean_sample:
            return True

        nepali_chars = len(re.findall(r"[\u0900-\u097F]", clean_sample))
        ratio = nepali_chars / len(clean_sample)

        # If more than 15% of characters are Nepali Unicode, we assume it's good.
        # (Nepali text usually has a high ratio of characters in this range).
        if ratio > 0.15:
            return True

        # If we have very few Nepali characters, check for legacy font indicators.
        # Pipe symbol '|' is frequently used as a danda '।' in legacy fonts.
        # Other common symbols: '÷', '×', '§', '°'
        legacy_markers = ["|", "÷", "§", "°", "¶"]
        has_legacy_markers = any(m in sample for m in legacy_markers)

        if ratio < 0.05 and has_legacy_markers:
            return False

        # If no Nepali characters at all, it might be English.
        # We check if there's any non-ASCII text.
        if nepali_chars == 0:
            # If it's purely ASCII, we assume English/Compatible and return True
            # unless it has legacy markers.
            return not has_legacy_markers

        # If ratio is low (e.g. 1-5%) and no legacy markers, it might be a
        # document with very little Nepali text. We'll be conservative.
        return ratio > 0.10

    def _extract_text_with_ocr(
        self, progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[str, str]:
        """
        Extract text and tables from the PDF using Tesseract OCR as a fallback.

        Returns:
            A tuple containing the main markdown content and the table markdown content.
            Note: Tesseract does not separate tables cleanly like PPStructure,
            so table_text will be empty or minimal, with content integrated into full_text.
        """
        logger.info("Falling back to Tesseract OCR for PDF: %s", self.pdf_path)
        if progress_callback:
            progress_callback(0, 1, "Initializing OCR engine...")
        try:
            # Check for pdftoppm (poppler)
            subprocess.run(["pdftoppm", "-h"], capture_output=True, check=False)
        except FileNotFoundError:
            logger.error(
                "Poppler (pdftoppm) not found. Please install poppler-utils to use the OCR fallback."
            )
            return "", ""

        # Convert PDF to images
        images = convert_from_path(self.pdf_path)

        full_text = []
        table_text = []  # Tesseract handles tables implicitly in text stream
        total_pages = len(images)

        for i, image in enumerate(images):
            if progress_callback:
                progress_callback(
                    i + 1,
                    total_pages,
                    f"Extracting text from page {i + 1} of {total_pages} using Tesseract...",
                )

            # Tesseract OCR
            # Using 'nep+eng' to support mixed language content
            try:
                # psm 3 is "Fully automatic page segmentation, but no OSD. (Default)"
                # which generally works best for full page scans including tables
                text = pytesseract.image_to_string(
                    image, lang="nep+eng", config="--psm 3"
                )

                full_text.append(f"<!-- page: {i + 1} -->")
                full_text.append(text)
                full_text.append("")
            except Exception as e:
                logger.error("Tesseract OCR failed for page %d: %s", i + 1, e)
                full_text.append(f"<!-- page: {i + 1} -->")
                full_text.append("[OCR Failed for this page]")
                full_text.append("")

        if progress_callback:
            progress_callback(total_pages, total_pages, "OCR Extraction complete.")

        return "\n".join(full_text), "\n".join(table_text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _extract_standard(
        self, progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> str:
        """Internal method for standard (non-OCR) extraction."""
        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)

        if progress_callback:
            progress_callback(0, total_pages, "Analyzing fonts...")

        body_size, heading_sizes = self._analyse_fonts(doc)

        page_lines: Dict[int, List[Dict[str, Any]]] = {}
        for page_num in range(total_pages):
            if progress_callback:
                progress_callback(
                    page_num + 1,
                    total_pages,
                    f"Extracting standard text from page {page_num + 1} of {total_pages}...",
                )
            page = doc[page_num]
            lines = self._extract_page_with_fonts(page, heading_sizes, body_size)
            page_lines[page_num + 1] = lines

        doc.close()

        if progress_callback:
            progress_callback(total_pages, total_pages, "Cleaning extracted text...")

        raw_md = self._lines_to_markdown(page_lines)
        return self._clean_text(raw_md)

    def extract_to_string(
        self, progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> str:
        """
        Extract PDF text and return the full Markdown as a string.

        If `use_ocr` is True, uses PaddleOCR for text extraction and PP-Structure for table extraction.
        Otherwise, uses standard PyMuPDF font-based extraction.
        Page markers (``<!-- page: N -->``) are embedded for downstream page-number
        mapping.

        Returns:
            Markdown string with text and page markers.
        """
        if self.use_ocr:
            logger.info(
                "PdfExtractor.extract_to_string  start (OCR)  pdf=%s", self.pdf_path
            )
            md_content, _ = self._extract_text_with_ocr(
                progress_callback=progress_callback
            )
        else:
            logger.info(
                "PdfExtractor.extract_to_string  start (Standard)  pdf=%s",
                self.pdf_path,
            )
            md_content = self._extract_standard(progress_callback=progress_callback)

        logger.info(
            "PdfExtractor.extract_to_string  done  chars=%d",
            len(md_content),
        )
        return md_content

    def extract(
        self,
        output_filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> str:
        """
        Extract PDF text and write the Markdown to a file.

        Args:
            output_filename: Override the output filename.
                             Defaults to ``<pdf_stem>.md``.

        Returns:
            Absolute path to the written .md file.
        """
        if output_filename is None:
            stem = Path(self.doc_name).stem
            output_filename = f"{stem}.md"
            table_output_filename = f"{stem}_tables.md"
        else:
            stem = Path(output_filename).stem
            table_output_filename = f"{stem}_tables.md"

        output_path = os.path.join(self.output_dir, output_filename)
        table_output_path = os.path.join(self.output_dir, table_output_filename)

        if self.use_ocr:
            logger.info("PdfExtractor.extract  processing %s (OCR)", self.pdf_path)
            md_content, table_content = self._extract_text_with_ocr(
                progress_callback=progress_callback
            )
        else:
            logger.info("PdfExtractor.extract  processing %s (Standard)", self.pdf_path)
            md_content = self._extract_standard(progress_callback=progress_callback)
            table_content = ""

        # 3. Save results
        os.makedirs(self.output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        if table_content:
            with open(table_output_path, "w", encoding="utf-8") as f:
                f.write(table_content)
            logger.info("PdfExtractor.extract  saved tables to %s", table_output_path)

        logger.info(
            "PdfExtractor.extract  done  path=%s",
            output_path,
        )
        return output_path


# ---------------------------------------------------------------------------
# Page-number mapping utility
# ---------------------------------------------------------------------------


def build_line_to_page_map(md_text: str) -> Dict[int, int]:
    """
    Build a mapping from 1-based line number to PDF page number.

    Scans the markdown for ``<!-- page: N -->`` markers and assigns
    all subsequent lines to that page until the next marker.

    Args:
        md_text: Markdown string produced by :class:`PdfExtractor`.

    Returns:
        Dict mapping line_number (1-based) to page_number (1-based).
    """
    import re

    page_re = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
    lines = md_text.split("\n")
    line_to_page: Dict[int, int] = {}
    current_page = 1

    for line_num, line in enumerate(lines, start=1):
        m = page_re.match(line.strip())
        if m:
            current_page = int(m.group(1))
        line_to_page[line_num] = current_page

    return line_to_page


def map_structure_pages(
    structure: List[dict],
    line_to_page: Dict[int, int],
    md_text: str,
) -> List[dict]:
    """
    Post-process a ``page_index_md`` structure tree to replace
    ``line_num`` with ``start_index`` / ``end_index`` (PDF page numbers).

    For each node:
      - ``start_index`` = page number of the node's ``line_num``
      - ``end_index`` = page number of the last line before the next
        sibling (or end of document)

    This makes the structure JSON compatible with the existing pipeline
    (TreeGrep, ContextFetcher, ContextMerger) which expect page numbers.

    Args:
        structure:    Structure tree from ``page_index_md.md_to_tree()``.
        line_to_page: Mapping from ``build_line_to_page_map()``.
        md_text:      The markdown text (to compute total lines).

    Returns:
        The structure tree, mutated in place, with ``start_index`` and
        ``end_index`` set on every node.
    """
    total_lines = len(md_text.split("\n"))

    def _collect_line_nums(nodes: List[dict]) -> List[int]:
        """Collect all line_num values from a flat list of sibling nodes."""
        nums = []
        for n in nodes:
            ln = n.get("line_num", 0)
            if ln > 0:
                nums.append(ln)
        return nums

    def _process_nodes(nodes: List[dict], end_line: int) -> None:
        """Recursively assign start_index / end_index to each node."""
        for i, node in enumerate(nodes):
            line_num = node.get("line_num", 0)

            # Determine the last line that belongs to this node:
            # it's the line just before the next sibling starts,
            # or end_line if this is the last sibling.
            if i + 1 < len(nodes):
                next_line = nodes[i + 1].get("line_num", end_line)
                node_end_line = next_line - 1
            else:
                node_end_line = end_line

            # Map to page numbers
            start_page = line_to_page.get(line_num, 1) if line_num > 0 else 1
            end_page = line_to_page.get(node_end_line, start_page)

            node["start_index"] = start_page
            node["end_index"] = end_page

            # Recurse into children
            children = node.get("nodes", [])
            if children:
                _process_nodes(children, node_end_line)

    _process_nodes(structure, total_lines)
    return structure


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def extract_pdf_to_markdown(
    pdf_path: str,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    use_ocr: bool = False,
) -> str:
    """
    Convenience function: extract PDF to Markdown in one call.

    Returns:
        Path to the written .md file.
    """
    extractor = PdfExtractor(
        pdf_path=pdf_path,
        output_dir=output_dir,
        use_ocr=use_ocr,
    )
    return extractor.extract(output_filename=output_filename)
