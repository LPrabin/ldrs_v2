"""
DocRegistry: Corpus-level inventory of all indexed documents.

Auto-generated from structure JSONs in a results directory. Produces a single
`_registry.json` that acts as a corpus-level Table of Contents. The LLM reads
this to decide which documents are relevant before drilling into any of them.

Registry entries contain:
  - doc_name, doc_description (if available)
  - index_path, md_path (if markdown cache exists)
  - top-level sections (titles only — lightweight TOC)
  - total node count, page range
  - indexed_at timestamp

The registry is rebuilt on every /index/batch call (cheap operation — just
reads JSON headers, no PDF work).
"""

import json
import logging
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry entry schema
# ---------------------------------------------------------------------------


def _count_nodes(nodes: List[dict]) -> int:
    """Recursively count all nodes in a structure tree."""
    total = 0
    for node in nodes:
        if node.get("node_id"):
            total += 1
        total += _count_nodes(node.get("nodes") or [])
    return total


def _page_range(nodes: List[dict]) -> tuple[int, int]:
    """Find the min start_index and max end_index across all nodes."""
    min_page = float("inf")
    max_page = 0
    for node in nodes:
        s = node.get("start_index", 0)
        e = node.get("end_index", 0)
        if s > 0:
            min_page = min(min_page, s)
        if e > 0:
            max_page = max(max_page, e)
        # Recurse into children
        if node.get("nodes"):
            child_min, child_max = _page_range(node["nodes"])
            if child_min < min_page:
                min_page = child_min
            if child_max > max_page:
                max_page = child_max
    if min_page == float("inf"):
        min_page = 0
    return int(min_page), int(max_page)


def _top_level_sections(structure: List[dict]) -> List[str]:
    """Extract titles of top-level nodes (the corpus TOC for this doc)."""
    sections = []
    for node in structure:
        title = node.get("title", "")
        if title:
            # NFC normalize for consistent display
            sections.append(unicodedata.normalize("NFC", title))
    return sections


def build_entry(
    index_path: str,
    md_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a single registry entry from a structure JSON file.

    Args:
        index_path: Path to the *_structure.json file.
        md_path:    Path to the cached .md file (None if not yet extracted).

    Returns:
        Dict with registry entry fields.
    """
    logger.debug("DocRegistry.build_entry  index_path=%s", index_path)

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    doc_name = index.get("doc_name", os.path.basename(index_path))
    doc_description = index.get("doc_description", "")
    structure = index.get("structure", [])

    node_count = _count_nodes(structure)
    min_page, max_page = _page_range(structure)
    sections = _top_level_sections(structure)

    entry: Dict[str, Any] = {
        "doc_name": doc_name,
        "doc_description": unicodedata.normalize("NFC", doc_description)
        if doc_description
        else "",
        "index_path": os.path.abspath(index_path),
        "md_path": os.path.abspath(md_path) if md_path else None,
        "node_count": node_count,
        "page_range": [min_page, max_page],
        "top_level_sections": sections,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.debug(
        "DocRegistry.build_entry  doc=%s  nodes=%d  pages=%d-%d  sections=%d",
        doc_name,
        node_count,
        min_page,
        max_page,
        len(sections),
    )
    return entry


# ---------------------------------------------------------------------------
# DocRegistry class
# ---------------------------------------------------------------------------


class DocRegistry:
    """
    Manages the corpus-level document registry.

    Usage:
        registry = DocRegistry(registry_path="results/_registry.json")
        registry.rebuild(results_dir="results/")   # scan all *_structure.json
        registry.save()

        # Or load an existing registry
        registry = DocRegistry.load("results/_registry.json")
        summary = registry.to_llm_summary()  # feed this to the doc selector LLM
    """

    REGISTRY_FILENAME = "_registry.json"

    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self.entries: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "version": "2.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.debug("DocRegistry init  path=%s", self.registry_path)

    @classmethod
    def load(cls, registry_path: str) -> "DocRegistry":
        """Load an existing registry from disk."""
        logger.info("DocRegistry.load  path=%s", registry_path)
        reg = cls(registry_path)
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            reg.entries = data.get("documents", [])
            reg.metadata = data.get("metadata", reg.metadata)
            logger.info("DocRegistry.load  loaded %d entries", len(reg.entries))
        else:
            logger.warning(
                "DocRegistry.load  file not found, starting empty: %s",
                registry_path,
            )
        return reg

    def save(self) -> str:
        """Write the registry to disk. Returns the path."""
        self.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        data = {
            "metadata": self.metadata,
            "documents": self.entries,
        }
        os.makedirs(os.path.dirname(self.registry_path) or ".", exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(
            "DocRegistry.save  written %d entries to %s",
            len(self.entries),
            self.registry_path,
        )
        return self.registry_path

    def rebuild(
        self,
        results_dir: str,
        md_dir: Optional[str] = None,
    ) -> int:
        """
        Rebuild the registry by scanning all *_structure.json files in a directory.

        Args:
            results_dir: Directory containing structure JSON files.
            md_dir:      Directory containing cached .md files (optional).
                         If None, md_path will be set to None for each entry.

        Returns:
            Number of documents registered.
        """
        logger.info(
            "DocRegistry.rebuild  results_dir=%s  md_dir=%s",
            results_dir,
            md_dir,
        )

        self.entries = []
        json_files = sorted(
            f for f in os.listdir(results_dir) if f.endswith("_structure.json")
        )

        logger.debug("DocRegistry.rebuild  found %d structure files", len(json_files))

        for json_file in json_files:
            index_path = os.path.join(results_dir, json_file)

            # Try to find a matching .md file
            md_path = None
            if md_dir:
                stem = json_file.replace("_structure.json", "")
                candidate = os.path.join(md_dir, f"{stem}.md")
                if os.path.exists(candidate):
                    md_path = candidate
                    logger.debug("DocRegistry.rebuild  found md cache: %s", candidate)

            try:
                entry = build_entry(index_path, md_path)
                self.entries.append(entry)
            except Exception as e:
                logger.error(
                    "DocRegistry.rebuild  failed to process %s: %s",
                    json_file,
                    e,
                )

        self.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(
            "DocRegistry.rebuild  done  registered %d documents",
            len(self.entries),
        )
        return len(self.entries)

    def add_or_update(
        self,
        index_path: str,
        md_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a single document to the registry, or update if it already exists.
        Matches on doc_name.

        Returns:
            The added/updated entry.
        """
        entry = build_entry(index_path, md_path)
        doc_name = entry["doc_name"]

        # Replace existing entry with same doc_name
        self.entries = [e for e in self.entries if e.get("doc_name") != doc_name]
        self.entries.append(entry)
        self.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            "DocRegistry.add_or_update  doc=%s  nodes=%d",
            doc_name,
            entry["node_count"],
        )
        return entry

    def remove(self, doc_name: str) -> bool:
        """
        Remove a document from the registry by name.

        Returns:
            True if the document was found and removed.
        """
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.get("doc_name") != doc_name]
        removed = len(self.entries) < before
        if removed:
            self.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
            logger.info("DocRegistry.remove  doc=%s  removed", doc_name)
        else:
            logger.warning("DocRegistry.remove  doc=%s  not found", doc_name)
        return removed

    def get_entry(self, doc_name: str) -> Optional[Dict[str, Any]]:
        """Look up a document entry by name."""
        for entry in self.entries:
            if entry.get("doc_name") == doc_name:
                return entry
        return None

    @property
    def doc_names(self) -> List[str]:
        """List all document names in the registry."""
        return [e.get("doc_name", "") for e in self.entries]

    def to_llm_summary(self) -> str:
        """
        Produce a concise, LLM-readable summary of the corpus.

        This string is designed to be injected into a doc-selector prompt
        so the LLM can decide which documents to search.
        """
        if not self.entries:
            return "The corpus is empty. No documents have been indexed."

        lines: List[str] = []
        lines.append(f"CORPUS REGISTRY — {len(self.entries)} document(s)\n")

        for i, entry in enumerate(self.entries, 1):
            doc_name = entry.get("doc_name", "?")
            desc = entry.get("doc_description", "")
            pages = entry.get("page_range", [0, 0])
            node_count = entry.get("node_count", 0)
            sections = entry.get("top_level_sections", [])

            lines.append(f"[{i}] {doc_name}")
            if desc:
                lines.append(f"    Description: {desc}")
            lines.append(f"    Pages: {pages[0]}-{pages[1]}  |  Nodes: {node_count}")
            if sections:
                section_str = " / ".join(sections[:10])
                if len(sections) > 10:
                    section_str += f" / ... (+{len(sections) - 10} more)"
                lines.append(f"    Sections: {section_str}")
            lines.append("")

        return "\n".join(lines)
