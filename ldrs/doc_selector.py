"""
DocSelector: LLM-based document selection from the corpus registry.

Given:
  - The corpus registry summary (from DocRegistry.to_llm_summary())
  - The corpus changelog summary (from ChangeLog.get_corpus_summary())
  - An expanded query (from QueryExpander)

The LLM decides which documents are most likely to contain relevant
information and returns their names + reasoning. This prevents wasting
tokens and time searching irrelevant documents.

Uses AsyncOpenAI (same client pattern as existing RAG modules).
"""

import json
import logging
import os
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

import openai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SELECTOR_SYSTEM_PROMPT = """\
You are a document selection assistant for a multi-document retrieval system.
You will be given:
1. A CORPUS REGISTRY listing all available documents with their descriptions,
   page ranges, and top-level section titles.
2. A CORPUS CHANGELOG showing recent additions, updates, and deletions.
3. A user query (possibly expanded into multiple sub-queries).

Your task: Select which documents are most likely to contain information
relevant to answering the query. Be inclusive — it's better to select a
document that might be relevant than to miss one that is.

Rules:
- Select at least 1 document (unless the corpus is empty).
- You may select ALL documents if the query is broad or ambiguous.
- Consider the changelog: recently deleted documents cannot be searched.
- Use section titles and descriptions to judge relevance.
- If the query is in a non-English language, match against document names
  and section titles in that language too.

Respond ONLY with valid JSON in this exact format:
{
  "selected_docs": ["doc_name_1.pdf", "doc_name_2.pdf"],
  "reasoning": "Brief explanation of why these documents were selected"
}

Do not include any text outside the JSON object."""

SELECTOR_USER_TEMPLATE = """\
=== CORPUS REGISTRY ===
{registry_summary}

=== CORPUS CHANGELOG ===
{changelog_summary}

=== QUERY ===
Original query: {original_query}

Sub-queries:
{sub_queries_text}

Select the documents most likely to contain relevant information."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DocSelection:
    """Result of document selection."""

    original_query: str
    selected_docs: List[str]
    reasoning: str = ""
    all_docs: List[str] = field(default_factory=list)

    @property
    def num_selected(self) -> int:
        return len(self.selected_docs)

    @property
    def num_total(self) -> int:
        return len(self.all_docs)


# ---------------------------------------------------------------------------
# DocSelector class
# ---------------------------------------------------------------------------


class DocSelector:
    """
    Select relevant documents from the corpus using an LLM.

    Usage:
        selector = DocSelector(model="qwen3-vl")
        selection = await selector.select(
            original_query="What is the revenue?",
            sub_queries=["total revenue", "revenue growth", "income statement"],
            registry_summary=registry.to_llm_summary(),
            changelog_summary=changelog.get_corpus_summary(),
            all_doc_names=registry.doc_names,
        )
        # selection.selected_docs -> ["financial_report.pdf"]
    """

    def __init__(self, model: str = "qwen3-vl"):
        self.model = model
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )
        logger.debug("DocSelector init  model=%s", self.model)

    async def select(
        self,
        original_query: str,
        sub_queries: List[str],
        registry_summary: str,
        changelog_summary: str,
        all_doc_names: List[str],
    ) -> DocSelection:
        """
        Ask the LLM to pick which documents to search.

        Args:
            original_query:    The user's original query string.
            sub_queries:       Expanded sub-queries (from QueryExpander).
            registry_summary:  Output of DocRegistry.to_llm_summary().
            changelog_summary: Output of ChangeLog.get_corpus_summary().
            all_doc_names:     List of all document names in the registry.

        Returns:
            DocSelection with selected doc names and reasoning.

        Fallback: if the LLM fails, returns ALL documents (safe default).
        """
        logger.info(
            "DocSelector.select  query=%r  sub_queries=%d  corpus_docs=%d",
            original_query,
            len(sub_queries),
            len(all_doc_names),
        )

        # Fast path: 0 or 1 documents — no need to ask the LLM
        if len(all_doc_names) == 0:
            logger.info("DocSelector.select  corpus is empty, nothing to select")
            return DocSelection(
                original_query=original_query,
                selected_docs=[],
                reasoning="Corpus is empty.",
                all_docs=[],
            )

        if len(all_doc_names) == 1:
            logger.info(
                "DocSelector.select  only 1 doc in corpus, auto-selecting: %s",
                all_doc_names[0],
            )
            return DocSelection(
                original_query=original_query,
                selected_docs=list(all_doc_names),
                reasoning="Only one document in corpus — auto-selected.",
                all_docs=list(all_doc_names),
            )

        # Build the sub-queries text
        sub_queries_text = "\n".join(
            f"  {i}. {q}" for i, q in enumerate(sub_queries, 1)
        )

        user_content = SELECTOR_USER_TEMPLATE.format(
            registry_summary=registry_summary,
            changelog_summary=changelog_summary or "No changelog available.",
            original_query=original_query,
            sub_queries_text=sub_queries_text,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
            )

            raw = response.choices[0].message.content or ""
            logger.debug("DocSelector  raw LLM response: %s", raw[:500])

            result = self._parse_response(raw, original_query, all_doc_names)
            logger.info(
                "DocSelector.select  done  selected=%d/%d  docs=%s  reasoning=%r",
                result.num_selected,
                result.num_total,
                result.selected_docs,
                result.reasoning[:100] if result.reasoning else "",
            )
            return result

        except Exception as e:
            logger.error("DocSelector.select  LLM call failed: %s", e)
            return DocSelection(
                original_query=original_query,
                selected_docs=list(all_doc_names),
                reasoning=f"Fallback: LLM selection failed ({e}), returning all documents.",
                all_docs=list(all_doc_names),
            )

    def _parse_response(
        self,
        raw: str,
        original_query: str,
        all_doc_names: List[str],
    ) -> DocSelection:
        """
        Parse the LLM JSON response into a DocSelection.

        Handles:
          - Markdown code fences around JSON
          - Extra text before/after JSON
          - Doc names that don't exist in the registry (filtered out)
          - NFC normalization for Nepali/Devanagari doc names
          - Empty selection (falls back to all)
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
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
                "DocSelector._parse_response  JSON parse failed: %s  raw=%r",
                e,
                raw[:200],
            )
            return DocSelection(
                original_query=original_query,
                selected_docs=list(all_doc_names),
                reasoning="JSON parse failed, returning all documents.",
                all_docs=list(all_doc_names),
            )

        selected_raw = data.get("selected_docs", [])
        reasoning = data.get("reasoning", "")

        if not isinstance(selected_raw, list):
            selected_raw = list(all_doc_names)

        if not isinstance(reasoning, str):
            reasoning = ""

        # NFC normalize all names for matching
        nfc_all = {unicodedata.normalize("NFC", n): n for n in all_doc_names}

        # Validate: keep only doc names that exist in the registry
        selected = []
        for name in selected_raw:
            if not isinstance(name, str):
                continue
            nfc_name = unicodedata.normalize("NFC", name.strip())
            if nfc_name in nfc_all:
                selected.append(nfc_all[nfc_name])
            else:
                # Try case-insensitive fuzzy match
                matched = self._fuzzy_match(nfc_name, nfc_all)
                if matched:
                    selected.append(matched)
                    logger.debug("DocSelector  fuzzy matched %r -> %r", name, matched)
                else:
                    logger.warning(
                        "DocSelector  LLM selected unknown doc: %r (skipped)",
                        name,
                    )

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for doc in selected:
            if doc not in seen:
                seen.add(doc)
                deduped.append(doc)
        selected = deduped

        # If LLM returned nothing valid, fall back to all docs
        if not selected:
            logger.warning(
                "DocSelector  LLM returned no valid docs, falling back to all"
            )
            selected = list(all_doc_names)
            reasoning = (reasoning or "") + " (Fallback: no valid docs selected)"

        return DocSelection(
            original_query=original_query,
            selected_docs=selected,
            reasoning=reasoning,
            all_docs=list(all_doc_names),
        )

    @staticmethod
    def _fuzzy_match(
        name: str,
        nfc_all: dict[str, str],
    ) -> Optional[str]:
        """
        Try case-insensitive match, then substring match.

        Returns the original (non-NFC-key) doc name if matched, else None.
        """
        name_lower = name.lower()

        # Exact case-insensitive
        for nfc_key, original in nfc_all.items():
            if nfc_key.lower() == name_lower:
                return original

        # Substring: if the LLM omitted .pdf extension or similar
        for nfc_key, original in nfc_all.items():
            if name_lower in nfc_key.lower() or nfc_key.lower() in name_lower:
                return original

        return None


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


async def select_documents(
    original_query: str,
    sub_queries: List[str],
    registry_summary: str,
    changelog_summary: str,
    all_doc_names: List[str],
    model: str = "qwen3-vl",
) -> DocSelection:
    """One-shot convenience function for document selection."""
    selector = DocSelector(model=model)
    return await selector.select(
        original_query=original_query,
        sub_queries=sub_queries,
        registry_summary=registry_summary,
        changelog_summary=changelog_summary,
        all_doc_names=all_doc_names,
    )
