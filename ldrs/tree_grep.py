"""
TreeGrep v2: Hierarchical pattern search across structure JSON and Markdown.

Major improvements over v1:
  - Searches **three** sources per node: title, summary (from JSON), and
    body text (from the cached .md file)
  - Full **NFC normalization** before every comparison — safe for
    Nepali/Devanagari text
  - Configurable **max_results** cap to prevent token explosion
  - Each GrepResult carries a **relevance_score** (title > summary > body)
  - Scope filtering by node_id or title (case-insensitive, NFC-aware)
  - Snippet extraction with configurable padding
  - Verbose DEBUG logging throughout
  - **Word-level tokenized matching** — natural-language sub-queries are
    split into individual words, stop words are filtered, and each word is
    matched independently.  Relevance scores are scaled by the fraction of
    matching words, so more-specific matches rank higher.
  - Regex matching uses ``re.IGNORECASE | re.MULTILINE | re.DOTALL`` flags.

Architecture:
  TreeGrep loads a structure JSON and (optionally) a companion .md file.
  When searching, it first walks the structure tree checking titles and
  summaries, then scans the Markdown body text keyed by node_id metadata
  comments (``<!-- node_id: XXXX | pages: X-X -->``).

Usage:
    grep = TreeGrep(index_path="doc_structure.json", md_path="doc.md")
    results = grep.search("inflation", max_results=20)
    for r in results:
        print(f"{r.breadcrumb} [{r.matched_field}] score={r.relevance_score}")
        print(f"  {r.snippet}")
"""

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Relevance weights — higher = more important.  Used for sorting results.
RELEVANCE_TITLE = 3.0
RELEVANCE_SUMMARY = 2.0
RELEVANCE_BODY = 1.0

# Default snippet padding (chars each side of match)
DEFAULT_SNIPPET_PADDING = 60

# Default max results returned
DEFAULT_MAX_RESULTS = 50

# Minimum fraction of content words that must match for a hit to count.
# E.g., if the sub-query has 5 content words and MIN_WORD_MATCH_RATIO=0.3,
# at least 2 words must appear in the text for it to be a hit.
MIN_WORD_MATCH_RATIO = 0.3

# Stop words — common English words that carry little search value.
# These are stripped from sub-queries before word-level matching so that
# "What is the Earth Movers Distance" becomes ["earth", "movers", "distance"].
STOP_WORDS = frozenset(
    {
        # Articles & determiners
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        # Pronouns
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "his",
        "her",
        "its",
        "they",
        "them",
        "their",
        # Prepositions
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "up",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "along",
        "until",
        "upon",
        "across",
        # Conjunctions
        "and",
        "but",
        "or",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        # Auxiliary / modal verbs
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        # Question words (useful to strip from sub-queries)
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        # Other common function words
        "not",
        "no",
        "yes",
        "if",
        "then",
        "else",
        "than",
        "also",
        "as",
        "just",
        "only",
        "very",
        "too",
        "more",
        "most",
        "some",
        "any",
        "all",
        "each",
        "every",
        "much",
        "many",
        "such",
        "own",
        # Common verbs that rarely add search specificity
        "get",
        "got",
        "make",
        "made",
        "take",
        "taken",
        "give",
        "given",
        "go",
        "gone",
        "come",
        "came",
        "say",
        "said",
        "tell",
        "told",
        "know",
        "known",
        "see",
        "seen",
        "find",
        "found",
        "use",
        "used",
        # Misc
        "there",
        "here",
        "other",
        "like",
        "well",
        "back",
    }
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GrepResult:
    """
    A single search hit inside the document tree.

    Attributes:
        node_id:         The node_id where the match was found.
        title:           The node's title.
        breadcrumb:      Ancestor path, e.g. ``"Chapter 1 > Section 2 > Subsection A"``.
        matched_field:   Which field matched: ``"title"``, ``"summary"``, or ``"body"``.
        snippet:         Text excerpt around the match.
        relevance_score: Numeric relevance (title=3, summary=2, body=1).
        page_range:      Tuple of ``(start_page, end_page)`` for the node (1-indexed).
    """

    node_id: str
    title: str
    breadcrumb: str
    matched_field: str
    snippet: str
    relevance_score: float = 1.0
    page_range: Tuple[int, int] = (0, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nfc(text: str) -> str:
    """Normalize a string to NFC form."""
    return unicodedata.normalize("NFC", text)


def _build_snippet(
    text: str, start: int, end: int, padding: int = DEFAULT_SNIPPET_PADDING
) -> str:
    """
    Extract a text snippet centred on ``text[start:end]`` with *padding*
    characters of context on each side.

    Args:
        text:    The full text to extract from.
        start:   Match start index within *text*.
        end:     Match end index within *text*.
        padding: Number of context characters on each side.

    Returns:
        Trimmed snippet string.  Leading/trailing whitespace collapsed.
    """
    left = max(0, start - padding)
    right = min(len(text), end + padding)
    snippet = text[left:right].strip()
    # Collapse runs of whitespace (common in PDF-extracted text)
    snippet = re.sub(r"\s+", " ", snippet)
    return snippet


def _tokenize_query(pattern: str) -> List[str]:
    """
    Tokenize a search pattern into individual content words.

    Splits on non-alphanumeric boundaries, lowercases, removes stop words,
    and discards tokens shorter than 2 characters.  Returns the remaining
    tokens in order.

    Examples::

        >>> _tokenize_query("What is Earth Movers Distance?")
        ['earth', 'movers', 'distance']
        >>> _tokenize_query("EMD algorithm")
        ['emd', 'algorithm']
        >>> _tokenize_query("optimal transport cost")
        ['optimal', 'transport', 'cost']

    Args:
        pattern: The raw search pattern (NFC-normalized).

    Returns:
        List of lowercase content words (stop words removed).
    """
    # Split on anything that is NOT an alphanumeric or Devanagari character.
    # The Unicode categories \w cover letters/digits/underscore; we also
    # keep Devanagari (U+0900–U+097F) via \w since Python \w is Unicode-aware.
    raw_tokens = re.split(r"[^\w]+", pattern.lower(), flags=re.UNICODE)
    tokens = [t for t in raw_tokens if t and len(t) >= 2 and t not in STOP_WORDS]
    return tokens


def _find_scope_nodes(nodes: List[dict], scope: Optional[str]) -> List[dict]:
    """
    Recursively locate subtree root(s) matching *scope*.

    A node matches if its ``node_id`` equals *scope* exactly **or** its
    ``title`` matches *scope* case-insensitively (NFC-normalized).

    Args:
        nodes: Top-level node list from the structure JSON.
        scope: node_id or title to filter by.  ``None`` → return all roots.

    Returns:
        List of matching nodes (may be empty).
    """
    if scope is None:
        return nodes

    scope_nfc = _nfc(scope).lower()
    matches: List[dict] = []

    for node in nodes:
        nid = node.get("node_id", "")
        title = _nfc(node.get("title", "")).lower()
        if nid == scope or title == scope_nfc:
            matches.append(node)
        children = node.get("nodes") or []
        if children:
            matches.extend(_find_scope_nodes(children, scope))

    return matches


# ---------------------------------------------------------------------------
# Markdown body parser
# ---------------------------------------------------------------------------


def _parse_md_sections(md_text: str) -> Dict[str, str]:
    """
    Parse a structured Markdown file (produced by ``MdExtractor``) into a
    mapping of ``{node_id: body_text}``.

    The .md format uses HTML comments as section markers::

        <!-- node_id: 0003 | pages: 5-7 -->

    Everything between one marker and the next (or EOF) is that node's body.

    Args:
        md_text: The full Markdown text.

    Returns:
        Dict mapping node_id strings to their body text.
    """
    sections: Dict[str, str] = {}
    marker_re = re.compile(r"<!--\s*node_id:\s*(\S+)\s*\|")

    parts = marker_re.split(md_text)
    # parts[0] is preamble before any marker
    # then alternating: node_id, body, node_id, body, ...
    idx = 1
    while idx < len(parts) - 1:
        node_id = parts[idx].strip()
        body = _nfc(parts[idx + 1].strip())
        sections[node_id] = body
        idx += 2

    logger.debug("_parse_md_sections  parsed %d sections from .md", len(sections))
    return sections


# ---------------------------------------------------------------------------
# TreeGrep class
# ---------------------------------------------------------------------------


class TreeGrep:
    """
    Hierarchical pattern search across a PageIndex structure tree and its
    companion Markdown file.

    The search checks three fields per node:

    1. **title** — from the structure JSON (relevance = 3.0)
    2. **summary** — from the structure JSON (relevance = 2.0)
    3. **body** — from the cached ``.md`` file (relevance = 1.0)

    Results are returned sorted by relevance (highest first), then by
    document order (node_id ascending).

    Args:
        index_path: Path to the ``*_structure.json`` file.
        md_path:    Path to the companion ``.md`` file.  If ``None``,
                    body-text search is skipped.

    Raises:
        FileNotFoundError: If *index_path* does not exist.

    Example::

        grep = TreeGrep("results/earthmover_structure.json", "results/earthmover.md")
        hits = grep.search("cost estimation")
        for h in hits:
            print(h.breadcrumb, h.snippet)
    """

    def __init__(
        self,
        index_path: str,
        md_path: Optional[str] = None,
    ):
        logger.info("TreeGrep init  index_path=%s  md_path=%s", index_path, md_path)

        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        self.structure: List[dict] = self.index.get("structure", [])
        self.doc_name: str = self.index.get("doc_name", os.path.basename(index_path))

        # Parse the companion .md file for body-text search
        self.md_sections: Dict[str, str] = {}
        if md_path and os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                self.md_sections = _parse_md_sections(f.read())
            logger.debug(
                "TreeGrep  loaded %d md sections from %s",
                len(self.md_sections),
                md_path,
            )
        elif md_path:
            logger.warning(
                "TreeGrep  md_path not found, body search disabled: %s", md_path
            )

        logger.debug(
            "TreeGrep  doc=%s  top_nodes=%d  md_sections=%d",
            self.doc_name,
            len(self.structure),
            len(self.md_sections),
        )

    # ------------------------------------------------------------------
    # Internal: recursive node search
    # ------------------------------------------------------------------

    def _search_node(
        self,
        node: dict,
        pattern_nfc: str,
        compiled_re: Optional[re.Pattern],
        content_tokens: List[str],
        word_regexes: List[re.Pattern],
        breadcrumb: List[str],
        results: List[GrepResult],
        use_regex: bool,
    ) -> None:
        """
        Recursively search a single node and its children.

        Checks title, summary, and body text.  Appends matches to *results*.

        Plain-text matching uses a two-tier strategy:

        1. **Exact substring match** — if the entire pattern appears as a
           contiguous substring, this is a perfect hit (score multiplier = 1.0).
        2. **Word-level match** — the pattern is tokenized (stop words removed).
           Each content word is searched independently.  If enough words match
           (≥ ``MIN_WORD_MATCH_RATIO``), the hit is accepted with its relevance
           score scaled by the fraction of matched words.

        This ensures both precise searches (e.g., ``"Earth Mover"``) and
        natural-language sub-queries (e.g., ``"What is Earth Movers Distance"``)
        produce results.

        Args:
            node:           The current structure node dict.
            pattern_nfc:    NFC-normalized search pattern (for plain-text mode).
            compiled_re:    Pre-compiled regex (for regex mode); ``None`` if not regex.
            content_tokens: Pre-tokenized content words from *pattern_nfc* (stop words removed).
            word_regexes:   Pre-compiled word-boundary regexes for each content token.
            breadcrumb:     List of ancestor titles (used to build breadcrumb string).
            results:        Accumulator list for matches.
            use_regex:      Whether to use regex matching.
        """
        title = _nfc(node.get("title", ""))
        summary = _nfc(node.get("summary", ""))
        node_id = node.get("node_id", "")
        start_idx = node.get("start_index", 0)
        end_idx = node.get("end_index", 0)
        crumb = " > ".join(breadcrumb + [title]) if title else " > ".join(breadcrumb)

        def _check(field_name: str, text: str, base_relevance: float) -> None:
            """
            Check one field against the pattern and append a GrepResult.

            For regex mode: uses compiled_re with IGNORECASE | MULTILINE | DOTALL.
            For plain-text mode:
              - First tries exact substring match (full score).
              - Falls back to word-level matching with score scaling.
            """
            if not text:
                return

            # ---- Regex mode ----
            if use_regex and compiled_re:
                match = compiled_re.search(text)
                if match:
                    snippet = _build_snippet(text, match.start(), match.end())
                    results.append(
                        GrepResult(
                            node_id=node_id,
                            title=title,
                            breadcrumb=crumb,
                            matched_field=field_name,
                            snippet=snippet,
                            relevance_score=base_relevance,
                            page_range=(start_idx, end_idx),
                        )
                    )
                return

            # ---- Plain-text mode ----
            text_lower = text.lower()
            target_lower = pattern_nfc.lower()

            # Tier 1: exact substring match → full relevance
            idx = text_lower.find(target_lower)
            if idx != -1:
                snippet = _build_snippet(text, idx, idx + len(target_lower))
                results.append(
                    GrepResult(
                        node_id=node_id,
                        title=title,
                        breadcrumb=crumb,
                        matched_field=field_name,
                        snippet=snippet,
                        relevance_score=base_relevance,
                        page_range=(start_idx, end_idx),
                    )
                )
                return  # exact match found — no need for word-level

            # Tier 2: word-level match
            if not content_tokens:
                return  # pattern was entirely stop words; nothing to match

            matched_count = 0
            first_match_start = len(text)
            first_match_end = 0

            for word_re in word_regexes:
                m = word_re.search(text)
                if m:
                    matched_count += 1
                    if m.start() < first_match_start:
                        first_match_start = m.start()
                        first_match_end = m.end()

            if matched_count == 0:
                return

            match_ratio = matched_count / len(content_tokens)
            if match_ratio < MIN_WORD_MATCH_RATIO:
                return

            # Scale relevance by the fraction of words that matched.
            # A single word out of 5 (ratio=0.2) is below threshold and
            # already filtered above.  3/5 (0.6) gets 60% of base score.
            scaled_relevance = base_relevance * match_ratio
            snippet = _build_snippet(text, first_match_start, first_match_end)
            results.append(
                GrepResult(
                    node_id=node_id,
                    title=title,
                    breadcrumb=crumb,
                    matched_field=field_name,
                    snippet=snippet,
                    relevance_score=round(scaled_relevance, 3),
                    page_range=(start_idx, end_idx),
                )
            )

        # Check title and summary from the structure JSON
        _check("title", title, RELEVANCE_TITLE)
        _check("summary", summary, RELEVANCE_SUMMARY)

        # Check body text from the .md file (if available)
        if node_id and node_id in self.md_sections:
            _check("body", self.md_sections[node_id], RELEVANCE_BODY)

        # Recurse into children
        for child in node.get("nodes") or []:
            self._search_node(
                child,
                pattern_nfc,
                compiled_re,
                content_tokens,
                word_regexes,
                breadcrumb + [title] if title else breadcrumb,
                results,
                use_regex,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        pattern: str,
        scope: Optional[str] = None,
        regex: bool = False,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> List[GrepResult]:
        """
        Search for *pattern* across the document tree.

        The search checks titles and summaries from the structure JSON,
        plus body text from the companion ``.md`` file.  Results are
        sorted by ``relevance_score`` (descending), then by ``node_id``
        (ascending, i.e. document order).

        **Plain-text mode** (default) uses two matching tiers:

        1. *Exact substring* — if the entire pattern appears verbatim in the
           text, this is a full-score hit.
        2. *Word-level* — the pattern is tokenized (stop words removed),
           each content word is searched independently via word-boundary
           regex, and the relevance score is scaled by the fraction of
           words that matched.  A minimum ratio (``MIN_WORD_MATCH_RATIO``)
           filters out noise.

        **Regex mode** uses ``re.IGNORECASE | re.MULTILINE | re.DOTALL``
        flags so patterns can span lines and match regardless of case.

        Args:
            pattern:     Search string or regex pattern.
            scope:       Optional node_id or title to restrict the search subtree.
            regex:       If ``True``, interpret *pattern* as a regex.
            max_results: Maximum number of results to return.

        Returns:
            List of :class:`GrepResult` objects, sorted by relevance then
            document order.  Empty list if *pattern* is empty.
        """
        logger.info(
            "TreeGrep.search  pattern=%r  scope=%s  regex=%s  max=%d",
            pattern,
            scope,
            regex,
            max_results,
        )

        if not pattern:
            logger.warning("TreeGrep.search  empty pattern, returning []")
            return []

        # NFC normalize the pattern
        pattern_nfc = _nfc(pattern)

        # Pre-compile regex if needed — with full flags
        compiled_re: Optional[re.Pattern] = None
        if regex:
            try:
                compiled_re = re.compile(
                    pattern_nfc,
                    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
                )
            except re.error as e:
                logger.error("TreeGrep.search  invalid regex %r: %s", pattern, e)
                return []

        # Tokenize pattern for word-level matching (plain-text mode only)
        content_tokens: List[str] = []
        word_regexes: List[re.Pattern] = []
        if not regex:
            content_tokens = _tokenize_query(pattern_nfc)
            # Build a case-insensitive regex for each content word.
            # We use loose matching (no word-boundary anchors) because
            # Devanagari and other scripts don't have \b boundaries, and
            # PDF-extracted text often has broken word boundaries.
            word_regexes = [
                re.compile(re.escape(tok), re.IGNORECASE | re.DOTALL)
                for tok in content_tokens
            ]
            logger.debug(
                "TreeGrep.search  content_tokens=%r  (from %d raw words)",
                content_tokens,
                len(re.split(r"[^\w]+", pattern_nfc, flags=re.UNICODE)),
            )

        # Find scope roots
        scope_roots = _find_scope_nodes(self.structure, scope)
        logger.debug("TreeGrep.search  scope_roots=%d", len(scope_roots))

        # Search
        results: List[GrepResult] = []
        for root in scope_roots:
            self._search_node(
                root,
                pattern_nfc,
                compiled_re,
                content_tokens,
                word_regexes,
                [],
                results,
                regex,
            )

        # Sort: highest relevance first, then by node_id (document order)
        results.sort(key=lambda r: (-r.relevance_score, r.node_id))

        # Cap
        if len(results) > max_results:
            logger.debug(
                "TreeGrep.search  capping results %d -> %d",
                len(results),
                max_results,
            )
            results = results[:max_results]

        logger.info(
            "TreeGrep.search  done  hits=%d  pattern=%r",
            len(results),
            pattern[:50],
        )
        return results

    def search_multi(
        self,
        patterns: List[str],
        scope: Optional[str] = None,
        regex: bool = False,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> List[GrepResult]:
        """
        Search for multiple patterns and merge results.

        Runs :meth:`search` for each pattern, deduplicates hits on the same
        ``(node_id, matched_field)`` pair (keeping the highest-scoring hit),
        then sorts and caps.

        This is designed to work with the sub-queries from
        :class:`QueryExpander`.

        Args:
            patterns:    List of search strings.
            scope:       Optional scope filter.
            regex:       Regex mode flag.
            max_results: Maximum total results.

        Returns:
            Merged, deduplicated, sorted list of :class:`GrepResult`.
        """
        logger.info(
            "TreeGrep.search_multi  patterns=%d  scope=%s  max=%d",
            len(patterns),
            scope,
            max_results,
        )

        # Collect all hits (allow more per-pattern, dedup later)
        per_pattern_cap = (
            max(max_results, max_results * 2 // len(patterns))
            if patterns
            else max_results
        )
        all_hits: List[GrepResult] = []
        for p in patterns:
            all_hits.extend(
                self.search(p, scope=scope, regex=regex, max_results=per_pattern_cap)
            )

        # Deduplicate: keep highest-scoring hit per (node_id, matched_field)
        best: Dict[Tuple[str, str], GrepResult] = {}
        for hit in all_hits:
            key = (hit.node_id, hit.matched_field)
            if key not in best or hit.relevance_score > best[key].relevance_score:
                best[key] = hit
        deduped = list(best.values())

        # Sort and cap
        deduped.sort(key=lambda r: (-r.relevance_score, r.node_id))
        if len(deduped) > max_results:
            deduped = deduped[:max_results]

        logger.info(
            "TreeGrep.search_multi  done  raw=%d  deduped=%d",
            len(all_hits),
            len(deduped),
        )
        return deduped
