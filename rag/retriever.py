"""
LLM-based Tree Search Retriever

This module provides the LLMRetriever class which uses an LLM to intelligently
select relevant document sections from a hierarchical tree structure.

The retriever works by:
1. Loading a hierarchical document index (TOC with page ranges)
2. Formatting the tree structure for the LLM
3. Sending a structured prompt to find relevant nodes
4. Parsing and validating the LLM's response
5. Returning nodes ordered by importance

Uses LLMProvider (centralized multi-provider LLM client via LiteLLM).

Example:
    retriever = LLMRetriever(
        index_path="results/test2_structure.json",
        model="qwen3-vl"
    )

    node_ids, reasoning = await retriever.retrieve(
        "What was the inflation rate in 2022-23?"
    )
    # Returns all relevant nodes, ordered by importance
"""

import json
import os
from typing import TYPE_CHECKING, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

if TYPE_CHECKING:
    from ldrs.llm_provider import LLMProvider


class LLMRetriever:
    """
    LLM-based tree search retriever for hierarchical document structures.

    Unlike traditional vector-based retrieval, this retriever uses an LLM
    to understand the document structure and select relevant sections based
    on titles, summaries, and the hierarchical relationships.

    The retriever returns all relevant nodes ordered by importance.
    The caller can then limit to top N for generation.

    Attributes:
        model: The LLM model to use for retrieval.
        llm_provider: LLMProvider instance for API calls.
        index: The loaded JSON index structure.
        node_map: Flattened mapping of node_id -> node data.

    Args:
        index_path: Path to the JSON index file.
        model: LLM model name (default: "qwen3-vl").

    Raises:
        FileNotFoundError: If the index file doesn't exist.
        KeyError: If required environment variables are missing.
    """

    def __init__(
        self,
        index_path: str,
        model: str = "qwen3-vl",
        llm_provider: "LLMProvider | None" = None,
    ):
        """
        Initialize the LLM retriever.

        Loads the index file and builds a node map for quick lookups.
        """
        self.model = model

        if llm_provider is not None:
            self.llm_provider = llm_provider
        else:
            from ldrs.llm_provider import get_provider

            self.llm_provider = get_provider()

        with open(index_path, "r") as f:
            self.index = json.load(f)

        self.node_map = self._build_node_map(self.index["structure"])

    def _build_node_map(
        self, nodes: List[dict], result: dict = None
    ) -> Dict[str, dict]:
        """
        Recursively flatten the tree into a node_id -> node mapping.

        This creates a quick lookup structure for validating node IDs
        returned by the LLM.

        Args:
            nodes: List of nodes to process.
            result: Accumulator dictionary (used for recursion).

        Returns:
            Dictionary mapping node_id to node data (title, start_index,
            end_index, summary).
        """
        if result is None:
            result = {}

        for node in nodes:
            if "node_id" in node:
                result[node["node_id"]] = {
                    "title": node.get("title", ""),
                    "start_index": node.get("start_index"),
                    "end_index": node.get("end_index"),
                    "summary": node.get("summary", ""),
                }
            if "nodes" in node:
                self._build_node_map(node["nodes"], result)

        return result

    def _get_tree_for_prompt(self) -> str:
        """
        Format the tree structure as a text representation for the LLM.

        Creates a hierarchical text representation with:
        - Node IDs in brackets
        - Indentation showing hierarchy
        - Truncated summaries (200 chars max)

        Example output:
            - [section_1] Overview
              Summary: This section provides an overview of...
            - [section_2] Details
              Summary: Detailed analysis of...

        Returns:
            Formatted tree string for the LLM prompt.
        """

        def format_node(node: dict, depth: int = 0) -> str:
            lines = []
            indent = "  " * depth
            node_id = node.get("node_id", "")
            title = node.get("title", "")
            summary = node.get("summary", "")[:200]  # Truncate long summaries

            if node_id:
                lines.append(f"{indent}- [{node_id}] {title}")
                if summary:
                    lines.append(f"{indent}  Summary: {summary}...")

            if "nodes" in node:
                for child in node["nodes"]:
                    lines.append(format_node(child, depth + 1))

            return "\n".join(lines)

        return "\n".join(format_node(n) for n in self.index["structure"])

    async def retrieve(self, query: str) -> tuple[List[str], str]:
        """
        Retrieve relevant node_ids for a given query.

        Sends a structured prompt to the LLM asking it to find ALL relevant
        nodes, ordered by importance/relevance. The LLM should rank nodes
        from most to least relevant.

        Args:
            query: The question to find relevant sections for.

        Returns:
            Tuple of (list of node_ids, reasoning string).
            The list is ordered by importance (most relevant first).
            The reasoning explains the ranking.

        Example:
            >>> node_ids, reasoning = await retriever.retrieve(
            ...     "What was the inflation rate?"
            ... )
            >>> print(node_ids)  # Ordered by importance
            ['section_3', 'subsection_3_1', 'section_5', 'section_10']
        """
        prompt = f"""You are given a question and a document structure.
Each node has [node_id] title and summary.
Find ALL nodes likely to contain the answer, ordered by importance.
Most relevant/relevant nodes should come first.

Question: {query}

Document Structure:
{self._get_tree_for_prompt()}

Reply in JSON format:
{{"thinking": "<explain which nodes are most relevant and why, ordered by importance>", "node_list": ["node_id_1", "node_id_2", "node_id_3"]}}

Return ONLY the JSON, nothing else. List ALL relevant nodes, ordered by importance (most relevant first)."""

        response = await self.llm_provider.acompletion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # Deterministic output for consistent retrieval
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON from response
        try:
            # Handle markdown code blocks (```json ... ```)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)
            node_ids = result.get("node_list", [])
            reasoning = result.get("thinking", "")

            # Validate node_ids exist in our index
            valid_ids = [nid for nid in node_ids if nid in self.node_map]

            return valid_ids, reasoning

        except json.JSONDecodeError:
            return [], "Failed to parse LLM response"
