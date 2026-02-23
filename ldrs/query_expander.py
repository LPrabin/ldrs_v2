"""
QueryExpander: LLM-powered multi-query expansion.

Takes a single user query and generates 3-5 sub-queries that cover different
angles, interpretations, and specificity levels. This improves recall by
ensuring downstream search hits more relevant sections across documents.

Uses LLMProvider (centralized multi-provider LLM client via LiteLLM).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv()

if TYPE_CHECKING:
    from ldrs.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

EXPANSION_SYSTEM_PROMPT = """\
You are a query expansion assistant for a document retrieval system.
Given a user's question, generate 3-5 sub-queries that approach the topic
from different angles. The sub-queries should:

1. Cover different aspects or facets of the original question
2. Use varied vocabulary (synonyms, related terms)
3. Range from broad to specific
4. Help find relevant information even if the original query wording
   doesn't match the document text exactly

Respond ONLY with valid JSON in this exact format:
{
  "sub_queries": ["query1", "query2", "query3"],
  "reasoning": "Brief explanation of why these sub-queries were chosen"
}

Do not include any text outside the JSON object."""

EXPANSION_USER_TEMPLATE = """\
Original query: {query}

Generate 3-5 sub-queries that cover different angles of this question."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExpandedQuery:
    """Result of query expansion."""

    original_query: str
    sub_queries: List[str]
    reasoning: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)  # prompt_tokens, completion_tokens, total_tokens, cost


# ---------------------------------------------------------------------------
# QueryExpander class
# ---------------------------------------------------------------------------


class QueryExpander:
    """
    Expand a single user query into multiple sub-queries using an LLM.

    Usage:
        expander = QueryExpander(model="qwen3-vl")
        result = await expander.expand("What is the company's revenue?")
        # result.sub_queries -> ["What is the total revenue...", "Revenue growth...", ...]

    The expanded queries are designed to maximize recall across a
    multi-document corpus by varying vocabulary, specificity, and angle.
    """

    def __init__(
        self,
        model: str = "qwen3-vl",
        max_sub_queries: int = 5,
        min_sub_queries: int = 3,
        llm_provider: Optional["LLMProvider"] = None,
    ):
        self.model = model
        self.max_sub_queries = max_sub_queries
        self.min_sub_queries = min_sub_queries

        if llm_provider is not None:
            self.llm_provider = llm_provider
        else:
            # Lazy import to avoid circular dependency at module level
            from ldrs.llm_provider import get_provider

            self.llm_provider = get_provider()

        logger.debug(
            "QueryExpander init  model=%s  provider=%s  max_sub=%d  min_sub=%d",
            self.model,
            self.llm_provider.provider_name,
            self.max_sub_queries,
            self.min_sub_queries,
        )

    async def expand(self, query: str) -> ExpandedQuery:
        """
        Expand a user query into multiple sub-queries.

        Args:
            query: The original user query.

        Returns:
            ExpandedQuery with original query, sub-queries, and reasoning.

        If the LLM call fails, returns a fallback with just the original query.
        """
        logger.info("QueryExpander.expand  query=%r", query)

        try:
            response = await self.llm_provider.acompletion(
                messages=[
                    {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": EXPANSION_USER_TEMPLATE.format(query=query),
                    },
                ],
                temperature=0.4,
            )

            raw = response.choices[0].message.content or ""
            usage = self.llm_provider.get_usage_and_cost(response)
            
            logger.debug("QueryExpander  raw LLM response: %s", raw[:500])

            result = self._parse_response(raw, query)
            result.usage = usage
            
            logger.info(
                "QueryExpander.expand  done  sub_queries=%d  reasoning=%r  cost=$%.6f",
                len(result.sub_queries),
                result.reasoning[:100] if result.reasoning else "",
                usage.get("cost", 0.0),
            )
            return result

        except Exception as e:
            logger.error("QueryExpander.expand  LLM call failed: %s", e)
            return ExpandedQuery(
                original_query=query,
                sub_queries=[query],
                reasoning=f"Fallback: LLM expansion failed ({e})",
            )

    def _parse_response(self, raw: str, original_query: str) -> ExpandedQuery:
        """
        Parse the LLM JSON response into an ExpandedQuery.

        Handles common LLM output issues:
          - Markdown code fences around JSON
          - Extra text before/after JSON
          - Missing fields
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (with optional language tag)
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Try to find JSON object in the response
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            cleaned = cleaned[json_start:json_end]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(
                "QueryExpander._parse_response  JSON parse failed: %s  raw=%r",
                e,
                raw[:200],
            )
            return ExpandedQuery(
                original_query=original_query,
                sub_queries=[original_query],
                reasoning=f"JSON parse failed, using original query",
            )

        sub_queries = data.get("sub_queries", [])
        reasoning = data.get("reasoning", "")

        # Validate and clamp
        if not isinstance(sub_queries, list):
            sub_queries = [original_query]

        # Filter out empty strings
        sub_queries = [
            q.strip() for q in sub_queries if isinstance(q, str) and q.strip()
        ]

        # Clamp to min/max
        if len(sub_queries) < self.min_sub_queries:
            # Pad with original query if too few
            while len(sub_queries) < self.min_sub_queries:
                sub_queries.append(original_query)
        elif len(sub_queries) > self.max_sub_queries:
            sub_queries = sub_queries[: self.max_sub_queries]

        return ExpandedQuery(
            original_query=original_query,
            sub_queries=sub_queries,
            reasoning=reasoning if isinstance(reasoning, str) else "",
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


async def expand_query(
    query: str,
    model: str = "qwen3-vl",
    llm_provider: Optional["LLMProvider"] = None,
) -> ExpandedQuery:
    """One-shot convenience function for query expansion."""
    expander = QueryExpander(model=model, llm_provider=llm_provider)
    return await expander.expand(query)
