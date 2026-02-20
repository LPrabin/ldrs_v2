"""
DocTelescope — Build contextual views of selected document-tree nodes.

Given a ``*_structure.json`` index and a list of ``node_id`` values, this
module constructs a *telescoped* view that includes each node's metadata,
its ancestor breadcrumb trail, its immediate children, sibling count, and
optionally the full extracted text from the source PDF.

This is useful for presenting *focused context* to an LLM or end-user:
rather than dumping the entire document, the telescope narrows the view
to exactly the nodes of interest and their structural neighbourhood.

Main API::

    telescope = DocTelescope(index_path="earthmover_structure.json",
                             pdf_path="earthmover.pdf")
    view = telescope.build(node_ids=["0003", "0007"], include_text=True)
    # view.nodes → list of enriched dicts
    # view.relationships → parent/child adjacency
    # view.traceability_ids → the input node_ids that were found

Classes:
    TelescopeView: Dataclass holding the build() result.
    DocTelescope:  Builder that loads an index and produces views.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
import logging

from rag.context_fetcher import ContextFetcher

logger = logging.getLogger(__name__)


@dataclass
class TelescopeView:
    """
    Result of :meth:`DocTelescope.build`.

    Attributes:
        nodes:            List of enriched node dicts, each containing
                          ``node_id``, ``title``, ``breadcrumb``, ``summary``,
                          ``children``, ``sibling_count``, ``start_index``,
                          ``end_index``, and optionally ``text``.
        relationships:    List of adjacency dicts (``node_id``, ``parent_id``,
                          ``children``) for graph-style traversal.
        traceability_ids: Ordered list of ``node_id`` values that were
                          successfully resolved from the input.
    """

    nodes: List[dict]
    relationships: List[dict]
    traceability_ids: List[str]


class DocTelescope:
    """
    Build structured previews for selected node IDs.
    """

    def __init__(self, index_path: str, pdf_path: Optional[str] = None):
        """
        Load a document index and build internal node/parent/children maps.

        Args:
            index_path: Path to the ``*_structure.json`` file.
            pdf_path:   Path to the source PDF (required only if
                        ``include_text=True`` in :meth:`build`).
        """
        logger.debug("DocTelescope loading index_path=%s", index_path)
        self.index_path = index_path
        with open(index_path, "r") as f:
            self.index = json.load(f)
        self.structure = self.index.get("structure", [])
        self.pdf_path = pdf_path
        self.node_map: Dict[str, dict] = {}
        self.parent_map: Dict[str, Optional[str]] = {}
        self.children_map: Dict[str, List[str]] = {}
        self._build_maps(self.structure, parent_id=None)

    def _build_maps(self, nodes: List[dict], parent_id: Optional[str]) -> None:
        """
        Recursively populate ``node_map``, ``parent_map``, and
        ``children_map`` from the structure tree.

        Args:
            nodes:     List of node dicts at the current depth.
            parent_id: ``node_id`` of the parent (``None`` for root level).
        """
        for node in nodes:
            node_id = node.get("node_id")
            if node_id:
                self.node_map[node_id] = node
                self.parent_map[node_id] = parent_id
                self.children_map.setdefault(node_id, [])

            for child in node.get("nodes", []) or []:
                child_id = child.get("node_id")
                if node_id and child_id:
                    self.children_map.setdefault(node_id, []).append(child_id)
                self._build_maps([child], parent_id=node_id if node_id else parent_id)

    def _breadcrumb(self, node_id: str) -> List[str]:
        """
        Build an ancestor title path from a leaf node up to the root.

        Returns:
            Titles in root-to-leaf order, e.g. ``["Chapter 1", "Section 2"]``.
        """
        breadcrumb = []
        current = node_id
        while current:
            node = self.node_map.get(current)
            if not node:
                break
            title = node.get("title", "")
            if title:
                breadcrumb.append(title)
            current = self.parent_map.get(current)
        return list(reversed(breadcrumb))

    def _sibling_count(self, node_id: str) -> int:
        """Return the number of siblings of *node_id* (excluding itself)."""
        parent = self.parent_map.get(node_id)
        if not parent:
            return 0
        return len(self.children_map.get(parent, [])) - 1

    def build(self, node_ids: List[str], include_text: bool = False) -> TelescopeView:
        """
        Build an enriched view for the given node IDs.

        For each *node_id*, the result includes the node's title, breadcrumb,
        summary, children, sibling count, and page range.  When
        ``include_text`` is ``True``, the extracted PDF body text for the
        node is also attached.

        Args:
            node_ids:     List of ``node_id`` strings to include.
            include_text: If ``True``, fetch body text via
                          :class:`ContextFetcher` (requires ``pdf_path``).

        Returns:
            A :class:`TelescopeView` with ``nodes``, ``relationships``,
            and ``traceability_ids``.

        Raises:
            ValueError: If ``include_text`` is ``True`` but ``pdf_path``
                        was not provided at construction time.
        """
        logger.info(
            "DocTelescope build start node_ids=%s include_text=%s",
            len(node_ids),
            include_text,
        )
        nodes: List[dict] = []
        relationships: List[dict] = []
        traceability_ids: List[str] = []

        context_fetcher = None
        if include_text:
            if not self.pdf_path:
                raise ValueError("pdf_path is required when include_text is True")
            context_fetcher = ContextFetcher(self.index_path, self.pdf_path)

        for node_id in node_ids:
            node = self.node_map.get(node_id)
            if not node:
                logger.warning("DocTelescope missing node_id=%s", node_id)
                continue
            breadcrumb = self._breadcrumb(node_id)
            children = []
            for child_id in self.children_map.get(node_id, []):
                child = self.node_map.get(child_id)
                if not child:
                    continue
                children.append(
                    {
                        "node_id": child_id,
                        "title": child.get("title", ""),
                        "summary": child.get("summary", ""),
                        "start_index": child.get("start_index"),
                        "end_index": child.get("end_index"),
                    }
                )

            entry: Dict[str, Any] = {
                "node_id": node_id,
                "title": node.get("title", ""),
                "breadcrumb": " > ".join(breadcrumb),
                "summary": node.get("summary", ""),
                "children": children,
                "sibling_count": self._sibling_count(node_id),
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
            }

            if include_text and context_fetcher:
                text_list = context_fetcher.fetch([node_id])
                entry["text"] = text_list[0] if text_list else ""

            nodes.append(entry)
            relationships.append(
                {
                    "node_id": node_id,
                    "parent_id": self.parent_map.get(node_id),
                    "children": [c.get("node_id") for c in children],
                }
            )
            traceability_ids.append(node_id)

        logger.info("DocTelescope build done nodes=%s", len(nodes))
        return TelescopeView(
            nodes=nodes, relationships=relationships, traceability_ids=traceability_ids
        )
