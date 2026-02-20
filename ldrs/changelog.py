"""
ChangeLog: Corpus file ledger for tracking document lifecycle.

This is NOT a query-history log. It tracks the lifecycle of documents
in the corpus:
  - INDEXED:  A new document was added to the corpus
  - UPDATED:  An existing document was re-indexed (structural diff recorded)
  - DELETED:  A document was removed from the corpus

Each entry carries:
  - commit_id:  SHA-256 hash of the structure JSON at that point in time
  - timestamp:  UTC ISO-8601
  - action:     indexed | updated | deleted
  - doc_name:   which document
  - diff:       structural changes (nodes added/removed/modified) — for updates

The LLM can read the changelog before searching to understand:
  - What documents are currently in the corpus
  - What's new or changed recently
  - What has been removed (avoid searching for deleted content)

Storage: append-only `_changelog.json` in the results directory.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural diff helpers
# ---------------------------------------------------------------------------


def _extract_node_set(structure: List[dict]) -> Dict[str, dict]:
    """
    Flatten a structure tree into {node_id: {title, start_index, end_index}}.
    Used for diffing two versions of the same document.
    """
    result: Dict[str, dict] = {}
    for node in structure:
        nid = node.get("node_id")
        if nid:
            result[nid] = {
                "title": node.get("title", ""),
                "start_index": node.get("start_index", 0),
                "end_index": node.get("end_index", 0),
            }
        for child in node.get("nodes") or []:
            result.update(_extract_node_set([child]))
    return result


def compute_structural_diff(
    old_structure: List[dict],
    new_structure: List[dict],
) -> Dict[str, Any]:
    """
    Compare two structure trees and return a diff.

    Returns:
        {
            "nodes_added":    [{"node_id": ..., "title": ...}, ...],
            "nodes_removed":  [{"node_id": ..., "title": ...}, ...],
            "nodes_modified": [{"node_id": ..., "old_title": ..., "new_title": ..., ...}, ...],
        }
    """
    old_nodes = _extract_node_set(old_structure)
    new_nodes = _extract_node_set(new_structure)

    old_ids: Set[str] = set(old_nodes.keys())
    new_ids: Set[str] = set(new_nodes.keys())

    added = []
    for nid in sorted(new_ids - old_ids):
        added.append({"node_id": nid, "title": new_nodes[nid]["title"]})

    removed = []
    for nid in sorted(old_ids - new_ids):
        removed.append({"node_id": nid, "title": old_nodes[nid]["title"]})

    modified = []
    for nid in sorted(old_ids & new_ids):
        old = old_nodes[nid]
        new = new_nodes[nid]
        changes: Dict[str, Any] = {}
        if old["title"] != new["title"]:
            changes["old_title"] = old["title"]
            changes["new_title"] = new["title"]
        if old["start_index"] != new["start_index"]:
            changes["old_start"] = old["start_index"]
            changes["new_start"] = new["start_index"]
        if old["end_index"] != new["end_index"]:
            changes["old_end"] = old["end_index"]
            changes["new_end"] = new["end_index"]
        if changes:
            changes["node_id"] = nid
            modified.append(changes)

    return {
        "nodes_added": added,
        "nodes_removed": removed,
        "nodes_modified": modified,
    }


def _compute_commit_id(index_data: Dict[str, Any]) -> str:
    """SHA-256 hash of the structure JSON content (deterministic)."""
    # Sort keys for deterministic hashing
    content = json.dumps(index_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# ChangeLog class
# ---------------------------------------------------------------------------


class ChangeLog:
    """
    Append-only ledger tracking document lifecycle in the corpus.

    Usage:
        log = ChangeLog(changelog_path="results/_changelog.json")
        log.record_indexed("report.pdf", index_data, structure)
        log.record_updated("report.pdf", index_data, old_structure, new_structure)
        log.record_deleted("report.pdf")
        log.save()

        # LLM context
        summary = log.get_corpus_summary()
    """

    CHANGELOG_FILENAME = "_changelog.json"

    def __init__(self, changelog_path: str):
        self.changelog_path = changelog_path
        self.entries: List[Dict[str, Any]] = []
        logger.debug("ChangeLog init  path=%s", self.changelog_path)

        # Load existing entries if file exists
        if os.path.exists(changelog_path):
            try:
                with open(changelog_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.entries = data.get("entries", [])
                logger.info("ChangeLog loaded %d existing entries", len(self.entries))
            except (json.JSONDecodeError, IOError) as e:
                logger.error("ChangeLog failed to load %s: %s", changelog_path, e)
                self.entries = []

    def save(self) -> str:
        """Write the changelog to disk. Returns the path."""
        data = {
            "version": "2.0",
            "entries": self.entries,
        }
        os.makedirs(os.path.dirname(self.changelog_path) or ".", exist_ok=True)
        with open(self.changelog_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(
            "ChangeLog.save  written %d entries to %s",
            len(self.entries),
            self.changelog_path,
        )
        return self.changelog_path

    def record_indexed(
        self,
        doc_name: str,
        index_data: Dict[str, Any],
        structure: List[dict],
    ) -> Dict[str, Any]:
        """
        Record that a new document was indexed (added to the corpus).

        Args:
            doc_name:   The document filename.
            index_data: Full index JSON (used for commit hash).
            structure:  The structure tree from the index.

        Returns:
            The changelog entry that was appended.
        """
        node_set = _extract_node_set(structure)
        commit_id = _compute_commit_id(index_data)

        entry: Dict[str, Any] = {
            "action": "indexed",
            "doc_name": doc_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "commit_id": commit_id,
            "node_count": len(node_set),
            "top_sections": [n.get("title", "") for n in structure if n.get("title")][
                :8
            ],
        }

        self.entries.append(entry)
        logger.info(
            "ChangeLog.record_indexed  doc=%s  commit=%s  nodes=%d",
            doc_name,
            commit_id,
            len(node_set),
        )
        return entry

    def record_updated(
        self,
        doc_name: str,
        index_data: Dict[str, Any],
        old_structure: List[dict],
        new_structure: List[dict],
    ) -> Dict[str, Any]:
        """
        Record that an existing document was re-indexed with changes.

        Args:
            doc_name:      The document filename.
            index_data:    New index JSON (for commit hash).
            old_structure: Previous structure tree.
            new_structure: New structure tree.

        Returns:
            The changelog entry that was appended.
        """
        diff = compute_structural_diff(old_structure, new_structure)
        commit_id = _compute_commit_id(index_data)

        entry: Dict[str, Any] = {
            "action": "updated",
            "doc_name": doc_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "commit_id": commit_id,
            "diff": diff,
        }

        self.entries.append(entry)
        logger.info(
            "ChangeLog.record_updated  doc=%s  commit=%s  "
            "added=%d  removed=%d  modified=%d",
            doc_name,
            commit_id,
            len(diff["nodes_added"]),
            len(diff["nodes_removed"]),
            len(diff["nodes_modified"]),
        )
        return entry

    def record_deleted(self, doc_name: str) -> Dict[str, Any]:
        """
        Record that a document was removed from the corpus.

        Returns:
            The changelog entry that was appended.
        """
        entry: Dict[str, Any] = {
            "action": "deleted",
            "doc_name": doc_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.entries.append(entry)
        logger.info("ChangeLog.record_deleted  doc=%s", doc_name)
        return entry

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_latest_action(self, doc_name: str) -> Optional[Dict[str, Any]]:
        """Get the most recent changelog entry for a document."""
        for entry in reversed(self.entries):
            if entry.get("doc_name") == doc_name:
                return entry
        return None

    def get_active_docs(self) -> List[str]:
        """
        Determine which documents are currently in the corpus.
        A document is active if its last action is 'indexed' or 'updated'
        (not 'deleted').
        """
        latest: Dict[str, str] = {}
        for entry in self.entries:
            doc = entry.get("doc_name", "")
            action = entry.get("action", "")
            if doc:
                latest[doc] = action

        return sorted(
            doc for doc, action in latest.items() if action in ("indexed", "updated")
        )

    def get_recent_changes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the N most recent changelog entries."""
        return list(reversed(self.entries[-n:]))

    def get_corpus_summary(self, recent_n: int = 5) -> str:
        """
        Produce an LLM-readable summary of the corpus state.

        Includes:
          - Currently active documents
          - Recent changes (last N entries)
          - Any deleted documents the LLM should know to skip
        """
        active = self.get_active_docs()
        recent = self.get_recent_changes(recent_n)

        lines: List[str] = []
        lines.append(f"CORPUS CHANGELOG — {len(self.entries)} total events\n")

        # Active documents
        lines.append(f"Active documents ({len(active)}):")
        if active:
            for doc in active:
                lines.append(f"  - {doc}")
        else:
            lines.append("  (none)")
        lines.append("")

        # Deleted documents
        deleted: set[str] = set()
        latest: Dict[str, str] = {}
        for entry in self.entries:
            doc = entry.get("doc_name", "")
            action = entry.get("action", "")
            if doc:
                latest[doc] = action
        deleted = {d for d, a in latest.items() if a == "deleted"}
        if deleted:
            lines.append(f"Deleted documents ({len(deleted)}):")
            for doc in sorted(deleted):
                lines.append(f"  - {doc} (removed, do not search)")
            lines.append("")

        # Recent changes
        if recent:
            lines.append(f"Recent changes (last {len(recent)}):")
            for entry in recent:
                action = entry.get("action", "?")
                doc = entry.get("doc_name", "?")
                ts = entry.get("timestamp", "?")
                # Truncate timestamp to just date+time
                ts_short = ts[:19] if len(ts) >= 19 else ts

                if action == "updated":
                    diff = entry.get("diff", {})
                    added = len(diff.get("nodes_added", []))
                    removed = len(diff.get("nodes_removed", []))
                    modified = len(diff.get("nodes_modified", []))
                    lines.append(
                        f"  [{ts_short}] {action.upper()} {doc}  "
                        f"(+{added} nodes, -{removed} nodes, ~{modified} modified)"
                    )
                else:
                    lines.append(f"  [{ts_short}] {action.upper()} {doc}")
            lines.append("")

        return "\n".join(lines)
