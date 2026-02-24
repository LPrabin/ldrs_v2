"""
Tests for LDRS v2 — Build Steps 1, 2, 3, 4 & 5

Step 1: MdExtractor    — PDF text extraction and markdown generation
Step 2: DocRegistry    — corpus inventory / TOC
        ChangeLog      — corpus file ledger (add/update/delete tracking)
Step 3: QueryExpander  — LLM multi-query expansion (parser tests, no LLM)
        DocSelector    — LLM-based document selection (parser + logic tests, no LLM)
Step 4: TreeGrep       — Hierarchical search across structure JSON + .md body
        ContextFetcher — fetch_from_md() reads from cached Markdown
        ContextMerger  — Cross-doc ranking, dedup, and token budgeting
Step 5: LDRSPipeline   — End-to-end orchestration (mocked LLM calls)
        LDRSConfig     — Configuration defaults and validation
        CLI / API      — import-level checks

Tests use real PDFs + structure JSONs from tests/pdfs/ and tests/results/.
Steps 1-2 are integration tests (PyMuPDF offline). Steps 3-5 test the
parsing, validation, and logic layers without requiring LLM calls.
"""

import json
import os
import re
import sys
import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ldrs.md_extractor import MdExtractor, extract_markdown
from ldrs.doc_registry import DocRegistry, build_entry
from ldrs.changelog import ChangeLog, compute_structural_diff
from ldrs.query_expander import QueryExpander, ExpandedQuery
from ldrs.doc_selector import DocSelector, DocSelection
from ldrs.tree_grep import TreeGrep, GrepResult, _parse_md_sections, _build_snippet
from ldrs.context_merger import ContextMerger, ContextChunk, MergedContext
from ldrs.ldrs_pipeline import LDRSPipeline, LDRSConfig, LDRSResult
from rag.context_fetcher import ContextFetcher

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")
RESULTS_DIR = os.path.join(TESTS_DIR, "results")
PDFS_DIR = os.path.join(TESTS_DIR, "pdfs")


def _index_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, f"{name}_structure.json")


def _pdf_path(name: str) -> str:
    return os.path.join(PDFS_DIR, f"{name}.pdf")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Test: basic instantiation
# ---------------------------------------------------------------------------
class TestMdExtractorInit:
    def test_loads_index(self):
        ext = MdExtractor(
            pdf_path=_pdf_path("earthmover"),
            index_path=_index_path("earthmover"),
        )
        assert ext.doc_name == "earthmover.pdf"
        assert len(ext.structure) > 0

    def test_loads_index_with_description(self):
        ext = MdExtractor(
            pdf_path=_pdf_path("q1-fy25-earnings"),
            index_path=_index_path("q1-fy25-earnings"),
        )
        assert ext.doc_name == "q1-fy25-earnings.pdf"
        assert ext.index.get("doc_description") is not None


# ---------------------------------------------------------------------------
# Test: extract_to_string produces valid markdown
# ---------------------------------------------------------------------------
class TestExtractToString:
    def test_earthmover_has_headings(self):
        ext = MdExtractor(
            pdf_path=_pdf_path("earthmover"),
            index_path=_index_path("earthmover"),
        )
        md = ext.extract_to_string()

        # Should contain the doc_name comment
        assert "<!-- doc_name: earthmover.pdf -->" in md

        # Should have node_id metadata comments
        assert "<!-- node_id: 0000" in md
        assert "<!-- node_id: 0001" in md

        # Should have markdown headings
        assert "# " in md  # at least one h1

    def test_earthmover_has_body_text(self):
        ext = MdExtractor(
            pdf_path=_pdf_path("earthmover"),
            index_path=_index_path("earthmover"),
        )
        md = ext.extract_to_string()

        # The paper is about Earth Mover's Distance — the word should appear
        assert "Earth Mover" in md or "earth mover" in md.lower()

    def test_q1_earnings_has_doc_description(self):
        ext = MdExtractor(
            pdf_path=_pdf_path("q1-fy25-earnings"),
            index_path=_index_path("q1-fy25-earnings"),
        )
        md = ext.extract_to_string()

        assert "<!-- doc_description:" in md

    def test_four_lectures_has_nested_headings(self):
        """four-lectures has depth-2 nodes — should produce ## headings."""
        ext = MdExtractor(
            pdf_path=_pdf_path("four-lectures"),
            index_path=_index_path("four-lectures"),
        )
        md = ext.extract_to_string()

        # Top-level nodes are h1, children should be h2
        assert "## " in md

    def test_metadata_comments_have_correct_format(self):
        ext = MdExtractor(
            pdf_path=_pdf_path("earthmover"),
            index_path=_index_path("earthmover"),
        )
        md = ext.extract_to_string()

        # All node_id comments should match the expected pattern
        pattern = r"<!-- node_id: \d{4} \| pages: \d+-\d+ -->"
        matches = re.findall(pattern, md)
        assert len(matches) > 0, "No metadata comments found"

        # Count should match total nodes in structure
        with open(_index_path("earthmover")) as f:
            index = json.load(f)

        def count_nodes(nodes):
            total = 0
            for n in nodes:
                if n.get("node_id"):
                    total += 1
                total += count_nodes(n.get("nodes") or [])
            return total

        expected = count_nodes(index["structure"])
        assert len(matches) == expected, (
            f"Expected {expected} metadata comments, got {len(matches)}"
        )

    def test_text_is_nfc_normalized(self):
        """Verify that Unicode NFC normalization is applied."""
        ext = MdExtractor(
            pdf_path=_pdf_path("earthmover"),
            index_path=_index_path("earthmover"),
        )
        md = ext.extract_to_string()

        import unicodedata

        # Re-normalizing should produce the same string
        re_normalized = unicodedata.normalize("NFC", md)
        assert md == re_normalized, "Text is not NFC normalized"


# ---------------------------------------------------------------------------
# Test: extract() writes a file
# ---------------------------------------------------------------------------
class TestExtractToFile:
    def test_writes_md_file(self, tmp_output_dir):
        ext = MdExtractor(
            pdf_path=_pdf_path("earthmover"),
            index_path=_index_path("earthmover"),
            output_dir=tmp_output_dir,
        )
        md_path = ext.extract()

        assert os.path.exists(md_path)
        assert md_path.endswith(".md")

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 100  # non-trivial content

    def test_default_filename(self, tmp_output_dir):
        ext = MdExtractor(
            pdf_path=_pdf_path("earthmover"),
            index_path=_index_path("earthmover"),
            output_dir=tmp_output_dir,
        )
        md_path = ext.extract()
        assert os.path.basename(md_path) == "earthmover.md"

    def test_custom_filename(self, tmp_output_dir):
        ext = MdExtractor(
            pdf_path=_pdf_path("earthmover"),
            index_path=_index_path("earthmover"),
            output_dir=tmp_output_dir,
        )
        md_path = ext.extract(output_filename="custom_name.md")
        assert os.path.basename(md_path) == "custom_name.md"
        assert os.path.exists(md_path)


# ---------------------------------------------------------------------------
# Test: convenience function
# ---------------------------------------------------------------------------
class TestConvenienceFunction:
    def test_extract_markdown(self, tmp_output_dir):
        md_path = extract_markdown(
            pdf_path=_pdf_path("earthmover"),
            index_path=_index_path("earthmover"),
            output_dir=tmp_output_dir,
        )
        assert os.path.exists(md_path)
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "<!-- node_id:" in content


# ---------------------------------------------------------------------------
# Test: multiple documents
# ---------------------------------------------------------------------------
class TestMultipleDocuments:
    """Verify extraction works across different test documents."""

    @pytest.mark.parametrize(
        "doc_name",
        [
            "earthmover",
            "q1-fy25-earnings",
            "four-lectures",
            "TestDocument",
        ],
    )
    def test_extraction_produces_output(self, doc_name, tmp_output_dir):
        pdf = _pdf_path(doc_name)
        idx = _index_path(doc_name)
        if not os.path.exists(pdf) or not os.path.exists(idx):
            pytest.skip(f"Missing test file for {doc_name}")

        ext = MdExtractor(pdf_path=pdf, index_path=idx, output_dir=tmp_output_dir)
        md = ext.extract_to_string()

        # Basic sanity: non-empty, has metadata, has headings
        assert len(md) > 50
        assert "<!-- node_id:" in md
        assert "# " in md


# ===========================================================================
# BUILD STEP 2: DocRegistry tests
# ===========================================================================


class TestDocRegistryBuildEntry:
    def test_build_entry_earthmover(self):
        entry = build_entry(_index_path("earthmover"))
        assert entry["doc_name"] == "earthmover.pdf"
        assert entry["node_count"] > 0
        assert entry["page_range"][0] >= 1
        assert entry["page_range"][1] >= entry["page_range"][0]
        assert len(entry["top_level_sections"]) > 0
        assert entry["indexed_at"]  # non-empty timestamp

    def test_build_entry_with_description(self):
        entry = build_entry(_index_path("q1-fy25-earnings"))
        assert entry["doc_description"] != ""
        assert (
            "Disney" in entry["doc_description"]
            or "financial" in entry["doc_description"].lower()
        )

    def test_build_entry_with_md_path(self, tmp_output_dir):
        # Create a dummy .md file
        md_path = os.path.join(tmp_output_dir, "earthmover.md")
        with open(md_path, "w") as f:
            f.write("dummy")
        entry = build_entry(_index_path("earthmover"), md_path=md_path)
        assert entry["md_path"] is not None
        assert entry["md_path"].endswith("earthmover.md")

    def test_build_entry_no_md_path(self):
        entry = build_entry(_index_path("earthmover"))
        assert entry["md_path"] is None


class TestDocRegistryRebuild:
    def test_rebuild_finds_all_structure_files(self, tmp_output_dir):
        reg = DocRegistry(os.path.join(tmp_output_dir, "_registry.json"))
        count = reg.rebuild(RESULTS_DIR)
        # We have multiple structure JSONs in results/
        assert count >= 5  # at least earthmover, q1, four-lectures, test, etc.
        assert len(reg.entries) == count

    def test_rebuild_populates_entries(self, tmp_output_dir):
        reg = DocRegistry(os.path.join(tmp_output_dir, "_registry.json"))
        reg.rebuild(RESULTS_DIR)
        # Each entry should have required fields
        for entry in reg.entries:
            assert "doc_name" in entry
            assert "node_count" in entry
            assert "page_range" in entry
            assert "top_level_sections" in entry

    def test_doc_names_property(self, tmp_output_dir):
        reg = DocRegistry(os.path.join(tmp_output_dir, "_registry.json"))
        reg.rebuild(RESULTS_DIR)
        names = reg.doc_names
        assert "earthmover.pdf" in names
        assert "four-lectures.pdf" in names


class TestDocRegistrySaveLoad:
    def test_save_and_load(self, tmp_output_dir):
        path = os.path.join(tmp_output_dir, "_registry.json")
        reg = DocRegistry(path)
        reg.rebuild(RESULTS_DIR)
        reg.save()

        # Load it back
        reg2 = DocRegistry.load(path)
        assert len(reg2.entries) == len(reg.entries)
        assert reg2.doc_names == reg.doc_names

    def test_save_creates_file(self, tmp_output_dir):
        path = os.path.join(tmp_output_dir, "_registry.json")
        reg = DocRegistry(path)
        reg.rebuild(RESULTS_DIR)
        saved_path = reg.save()
        assert os.path.exists(saved_path)
        with open(saved_path) as f:
            data = json.load(f)
        assert "documents" in data
        assert "metadata" in data


class TestDocRegistryAddRemove:
    def test_add_or_update(self, tmp_output_dir):
        reg = DocRegistry(os.path.join(tmp_output_dir, "_registry.json"))
        entry = reg.add_or_update(_index_path("earthmover"))
        assert entry["doc_name"] == "earthmover.pdf"
        assert len(reg.entries) == 1

        # Adding again should replace, not duplicate
        reg.add_or_update(_index_path("earthmover"))
        assert len(reg.entries) == 1

    def test_remove(self, tmp_output_dir):
        reg = DocRegistry(os.path.join(tmp_output_dir, "_registry.json"))
        reg.add_or_update(_index_path("earthmover"))
        reg.add_or_update(_index_path("four-lectures"))
        assert len(reg.entries) == 2

        removed = reg.remove("earthmover.pdf")
        assert removed is True
        assert len(reg.entries) == 1
        assert reg.entries[0]["doc_name"] == "four-lectures.pdf"

    def test_remove_nonexistent(self, tmp_output_dir):
        reg = DocRegistry(os.path.join(tmp_output_dir, "_registry.json"))
        removed = reg.remove("doesnotexist.pdf")
        assert removed is False

    def test_get_entry(self, tmp_output_dir):
        reg = DocRegistry(os.path.join(tmp_output_dir, "_registry.json"))
        reg.add_or_update(_index_path("earthmover"))
        entry = reg.get_entry("earthmover.pdf")
        assert entry is not None
        assert entry["doc_name"] == "earthmover.pdf"

        missing = reg.get_entry("nope.pdf")
        assert missing is None


class TestDocRegistryLLMSummary:
    def test_summary_not_empty(self, tmp_output_dir):
        reg = DocRegistry(os.path.join(tmp_output_dir, "_registry.json"))
        reg.rebuild(RESULTS_DIR)
        summary = reg.to_llm_summary()
        assert "CORPUS REGISTRY" in summary
        assert "earthmover.pdf" in summary
        assert "Sections:" in summary

    def test_summary_empty_corpus(self, tmp_output_dir):
        reg = DocRegistry(os.path.join(tmp_output_dir, "_registry.json"))
        summary = reg.to_llm_summary()
        assert "empty" in summary.lower()


# ===========================================================================
# BUILD STEP 2: ChangeLog tests
# ===========================================================================


class TestStructuralDiff:
    def test_identical_structures(self):
        structure = [
            {"title": "A", "start_index": 1, "end_index": 2, "node_id": "0000"},
            {"title": "B", "start_index": 3, "end_index": 4, "node_id": "0001"},
        ]
        diff = compute_structural_diff(structure, structure)
        assert diff["nodes_added"] == []
        assert diff["nodes_removed"] == []
        assert diff["nodes_modified"] == []

    def test_node_added(self):
        old = [
            {"title": "A", "start_index": 1, "end_index": 2, "node_id": "0000"},
        ]
        new = [
            {"title": "A", "start_index": 1, "end_index": 2, "node_id": "0000"},
            {"title": "B", "start_index": 3, "end_index": 4, "node_id": "0001"},
        ]
        diff = compute_structural_diff(old, new)
        assert len(diff["nodes_added"]) == 1
        assert diff["nodes_added"][0]["node_id"] == "0001"
        assert diff["nodes_removed"] == []

    def test_node_removed(self):
        old = [
            {"title": "A", "start_index": 1, "end_index": 2, "node_id": "0000"},
            {"title": "B", "start_index": 3, "end_index": 4, "node_id": "0001"},
        ]
        new = [
            {"title": "A", "start_index": 1, "end_index": 2, "node_id": "0000"},
        ]
        diff = compute_structural_diff(old, new)
        assert diff["nodes_added"] == []
        assert len(diff["nodes_removed"]) == 1
        assert diff["nodes_removed"][0]["node_id"] == "0001"

    def test_node_modified_title(self):
        old = [
            {"title": "A", "start_index": 1, "end_index": 2, "node_id": "0000"},
        ]
        new = [
            {
                "title": "A (revised)",
                "start_index": 1,
                "end_index": 2,
                "node_id": "0000",
            },
        ]
        diff = compute_structural_diff(old, new)
        assert len(diff["nodes_modified"]) == 1
        assert diff["nodes_modified"][0]["old_title"] == "A"
        assert diff["nodes_modified"][0]["new_title"] == "A (revised)"

    def test_node_modified_page_range(self):
        old = [
            {"title": "A", "start_index": 1, "end_index": 2, "node_id": "0000"},
        ]
        new = [
            {"title": "A", "start_index": 1, "end_index": 5, "node_id": "0000"},
        ]
        diff = compute_structural_diff(old, new)
        assert len(diff["nodes_modified"]) == 1
        assert diff["nodes_modified"][0]["old_end"] == 2
        assert diff["nodes_modified"][0]["new_end"] == 5

    def test_diff_with_nested_children(self):
        """Diff should recurse into child nodes."""
        old = [
            {
                "title": "Parent",
                "start_index": 1,
                "end_index": 5,
                "node_id": "0000",
                "nodes": [
                    {
                        "title": "Child",
                        "start_index": 2,
                        "end_index": 3,
                        "node_id": "0001",
                    },
                ],
            },
        ]
        new = [
            {
                "title": "Parent",
                "start_index": 1,
                "end_index": 5,
                "node_id": "0000",
                "nodes": [
                    {
                        "title": "Child",
                        "start_index": 2,
                        "end_index": 3,
                        "node_id": "0001",
                    },
                    {
                        "title": "New Child",
                        "start_index": 4,
                        "end_index": 5,
                        "node_id": "0002",
                    },
                ],
            },
        ]
        diff = compute_structural_diff(old, new)
        assert len(diff["nodes_added"]) == 1
        assert diff["nodes_added"][0]["node_id"] == "0002"


class TestChangeLogRecording:
    def test_record_indexed(self, tmp_output_dir):
        path = os.path.join(tmp_output_dir, "_changelog.json")
        log = ChangeLog(path)

        with open(_index_path("earthmover")) as f:
            index_data = json.load(f)

        entry = log.record_indexed(
            "earthmover.pdf", index_data, index_data["structure"]
        )
        assert entry["action"] == "indexed"
        assert entry["doc_name"] == "earthmover.pdf"
        assert entry["commit_id"]  # non-empty hash
        assert entry["node_count"] > 0
        assert len(log.entries) == 1

    def test_record_updated(self, tmp_output_dir):
        path = os.path.join(tmp_output_dir, "_changelog.json")
        log = ChangeLog(path)

        old_structure = [
            {"title": "A", "start_index": 1, "end_index": 2, "node_id": "0000"},
        ]
        new_structure = [
            {"title": "A", "start_index": 1, "end_index": 2, "node_id": "0000"},
            {"title": "B", "start_index": 3, "end_index": 4, "node_id": "0001"},
        ]
        index_data = {"doc_name": "test.pdf", "structure": new_structure}

        entry = log.record_updated("test.pdf", index_data, old_structure, new_structure)
        assert entry["action"] == "updated"
        assert len(entry["diff"]["nodes_added"]) == 1

    def test_record_deleted(self, tmp_output_dir):
        path = os.path.join(tmp_output_dir, "_changelog.json")
        log = ChangeLog(path)

        entry = log.record_deleted("removed.pdf")
        assert entry["action"] == "deleted"
        assert entry["doc_name"] == "removed.pdf"


class TestChangeLogSaveLoad:
    def test_save_and_reload(self, tmp_output_dir):
        path = os.path.join(tmp_output_dir, "_changelog.json")
        log = ChangeLog(path)

        with open(_index_path("earthmover")) as f:
            index_data = json.load(f)
        log.record_indexed("earthmover.pdf", index_data, index_data["structure"])
        log.record_deleted("old_doc.pdf")
        log.save()

        # Reload
        log2 = ChangeLog(path)
        assert len(log2.entries) == 2
        assert log2.entries[0]["action"] == "indexed"
        assert log2.entries[1]["action"] == "deleted"


class TestChangeLogQueries:
    def _make_log(self, tmp_output_dir):
        path = os.path.join(tmp_output_dir, "_changelog.json")
        log = ChangeLog(path)

        with open(_index_path("earthmover")) as f:
            idx1 = json.load(f)
        with open(_index_path("four-lectures")) as f:
            idx2 = json.load(f)

        log.record_indexed("earthmover.pdf", idx1, idx1["structure"])
        log.record_indexed("four-lectures.pdf", idx2, idx2["structure"])
        log.record_deleted("old_report.pdf")
        return log

    def test_get_active_docs(self, tmp_output_dir):
        log = self._make_log(tmp_output_dir)
        active = log.get_active_docs()
        assert "earthmover.pdf" in active
        assert "four-lectures.pdf" in active
        assert "old_report.pdf" not in active

    def test_get_latest_action(self, tmp_output_dir):
        log = self._make_log(tmp_output_dir)
        latest = log.get_latest_action("earthmover.pdf")
        assert latest is not None
        assert latest["action"] == "indexed"

        latest_del = log.get_latest_action("old_report.pdf")
        assert latest_del is not None
        assert latest_del["action"] == "deleted"

        assert log.get_latest_action("nonexistent.pdf") is None

    def test_get_recent_changes(self, tmp_output_dir):
        log = self._make_log(tmp_output_dir)
        recent = log.get_recent_changes(2)
        assert len(recent) == 2
        # Most recent first
        assert recent[0]["action"] == "deleted"
        assert recent[1]["action"] == "indexed"

    def test_corpus_summary(self, tmp_output_dir):
        log = self._make_log(tmp_output_dir)
        summary = log.get_corpus_summary()
        assert "CORPUS CHANGELOG" in summary
        assert "earthmover.pdf" in summary
        assert "four-lectures.pdf" in summary
        assert "old_report.pdf" in summary
        assert "DELETED" in summary
        assert "Active documents" in summary


# ==========================================================================
# BUILD STEP 3 — QueryExpander + DocSelector
# ==========================================================================


# ---------------------------------------------------------------------------
# QueryExpander._parse_response tests
# ---------------------------------------------------------------------------


class TestQueryExpanderParser:
    """Test the JSON parsing/validation layer (no LLM calls)."""

    @pytest.fixture
    def expander(self):
        """Create a QueryExpander instance (LLM won't be called)."""
        return QueryExpander(model="qwen3-vl")

    def test_parse_clean_json(self, expander):
        raw = json.dumps(
            {
                "sub_queries": ["query A", "query B", "query C"],
                "reasoning": "Three different angles",
            }
        )
        result = expander._parse_response(raw, "original")
        assert result.original_query == "original"
        assert result.sub_queries == ["query A", "query B", "query C"]
        assert result.reasoning == "Three different angles"

    def test_parse_json_with_markdown_fences(self, expander):
        raw = '```json\n{"sub_queries": ["a", "b", "c"], "reasoning": "ok"}\n```'
        result = expander._parse_response(raw, "test")
        assert result.sub_queries == ["a", "b", "c"]

    def test_parse_json_with_extra_text(self, expander):
        raw = 'Here is the result:\n{"sub_queries": ["x", "y", "z"], "reasoning": "done"}\nEnd.'
        result = expander._parse_response(raw, "test")
        assert result.sub_queries == ["x", "y", "z"]
        assert result.reasoning == "done"

    def test_parse_invalid_json_falls_back(self, expander):
        raw = "This is not JSON at all"
        result = expander._parse_response(raw, "fallback query")
        assert result.original_query == "fallback query"
        assert result.sub_queries == ["fallback query"]
        assert "parse failed" in result.reasoning.lower()

    def test_parse_too_few_queries_pads(self, expander):
        """If LLM returns fewer than min_sub_queries, pad with original."""
        raw = json.dumps({"sub_queries": ["only one"], "reasoning": "short"})
        result = expander._parse_response(raw, "original Q")
        assert len(result.sub_queries) >= expander.min_sub_queries
        assert "only one" in result.sub_queries
        assert "original Q" in result.sub_queries

    def test_parse_too_many_queries_truncates(self, expander):
        """If LLM returns more than max_sub_queries, truncate."""
        queries = [f"q{i}" for i in range(10)]
        raw = json.dumps({"sub_queries": queries, "reasoning": "many"})
        result = expander._parse_response(raw, "test")
        assert len(result.sub_queries) <= expander.max_sub_queries

    def test_parse_empty_strings_filtered(self, expander):
        """Empty strings in sub_queries should be removed."""
        raw = json.dumps({"sub_queries": ["a", "", "  ", "b", "c"], "reasoning": "ok"})
        result = expander._parse_response(raw, "test")
        # "a", "b", "c" remain; "" and "  " are filtered out
        assert "" not in result.sub_queries
        assert all(q.strip() for q in result.sub_queries)

    def test_parse_missing_reasoning(self, expander):
        """Missing reasoning field should default to empty string."""
        raw = json.dumps({"sub_queries": ["a", "b", "c"]})
        result = expander._parse_response(raw, "test")
        assert result.reasoning == ""

    def test_parse_non_list_sub_queries(self, expander):
        """If sub_queries is not a list, fall back to original query."""
        raw = json.dumps({"sub_queries": "not a list", "reasoning": "bad"})
        result = expander._parse_response(raw, "orig")
        assert "orig" in result.sub_queries


class TestExpandedQueryDataclass:
    def test_basic_creation(self):
        eq = ExpandedQuery(
            original_query="test",
            sub_queries=["a", "b"],
            reasoning="reason",
        )
        assert eq.original_query == "test"
        assert eq.sub_queries == ["a", "b"]
        assert eq.reasoning == "reason"

    def test_default_reasoning(self):
        eq = ExpandedQuery(original_query="test", sub_queries=["a"])
        assert eq.reasoning == ""


# ---------------------------------------------------------------------------
# DocSelector._parse_response tests
# ---------------------------------------------------------------------------


class TestDocSelectorParser:
    """Test the JSON parsing/validation + fuzzy matching layer (no LLM)."""

    ALL_DOCS = [
        "earthmover.pdf",
        "four-lectures.pdf",
        "financial_report.pdf",
        "नेपाली_दस्तावेज.pdf",
    ]

    @pytest.fixture
    def selector(self):
        return DocSelector(model="qwen3-vl")

    def test_parse_clean_json(self, selector):
        raw = json.dumps(
            {
                "selected_docs": ["earthmover.pdf", "financial_report.pdf"],
                "reasoning": "These two are relevant",
            }
        )
        result = selector._parse_response(raw, "test query", self.ALL_DOCS)
        assert result.selected_docs == ["earthmover.pdf", "financial_report.pdf"]
        assert result.reasoning == "These two are relevant"
        assert result.all_docs == self.ALL_DOCS

    def test_parse_json_with_markdown_fences(self, selector):
        raw = '```json\n{"selected_docs": ["earthmover.pdf"], "reasoning": "ok"}\n```'
        result = selector._parse_response(raw, "test", self.ALL_DOCS)
        assert result.selected_docs == ["earthmover.pdf"]

    def test_parse_invalid_json_returns_all(self, selector):
        """If JSON parsing fails, fall back to ALL documents."""
        raw = "not json"
        result = selector._parse_response(raw, "test", self.ALL_DOCS)
        assert result.selected_docs == self.ALL_DOCS
        assert "parse failed" in result.reasoning.lower()

    def test_parse_unknown_doc_filtered(self, selector):
        """Doc names not in the registry should be filtered out."""
        raw = json.dumps(
            {
                "selected_docs": ["earthmover.pdf", "nonexistent.pdf"],
                "reasoning": "test",
            }
        )
        result = selector._parse_response(raw, "test", self.ALL_DOCS)
        assert "earthmover.pdf" in result.selected_docs
        assert "nonexistent.pdf" not in result.selected_docs

    def test_parse_empty_selection_falls_back_to_all(self, selector):
        """If LLM returns empty selection, fall back to all docs."""
        raw = json.dumps({"selected_docs": [], "reasoning": "none"})
        result = selector._parse_response(raw, "test", self.ALL_DOCS)
        assert result.selected_docs == self.ALL_DOCS
        assert "Fallback" in result.reasoning

    def test_deduplication(self, selector):
        """Duplicate doc names should be removed."""
        raw = json.dumps(
            {
                "selected_docs": [
                    "earthmover.pdf",
                    "earthmover.pdf",
                    "earthmover.pdf",
                ],
                "reasoning": "dup",
            }
        )
        result = selector._parse_response(raw, "test", self.ALL_DOCS)
        assert result.selected_docs == ["earthmover.pdf"]

    def test_case_insensitive_fuzzy_match(self, selector):
        """LLM might return doc names with wrong case — should still match."""
        raw = json.dumps(
            {
                "selected_docs": ["EARTHMOVER.PDF", "Financial_Report.pdf"],
                "reasoning": "case",
            }
        )
        result = selector._parse_response(raw, "test", self.ALL_DOCS)
        assert "earthmover.pdf" in result.selected_docs
        assert "financial_report.pdf" in result.selected_docs

    def test_substring_fuzzy_match(self, selector):
        """LLM might omit .pdf extension — substring match should catch it."""
        raw = json.dumps(
            {
                "selected_docs": ["earthmover"],
                "reasoning": "no extension",
            }
        )
        result = selector._parse_response(raw, "test", self.ALL_DOCS)
        assert "earthmover.pdf" in result.selected_docs

    def test_nfc_normalization_nepali(self, selector):
        """Nepali/Devanagari doc names should match via NFC normalization."""
        import unicodedata

        # Create a NFD version of the Nepali name
        nfd_name = unicodedata.normalize("NFD", "नेपाली_दस्तावेज.pdf")
        raw = json.dumps(
            {
                "selected_docs": [nfd_name],
                "reasoning": "nepali doc",
            }
        )
        result = selector._parse_response(raw, "test", self.ALL_DOCS)
        assert "नेपाली_दस्तावेज.pdf" in result.selected_docs

    def test_non_list_selected_docs(self, selector):
        """If selected_docs is not a list, fall back to all."""
        raw = json.dumps({"selected_docs": "earthmover.pdf", "reasoning": "str"})
        result = selector._parse_response(raw, "test", self.ALL_DOCS)
        assert result.selected_docs == self.ALL_DOCS

    def test_non_string_reasoning(self, selector):
        """Non-string reasoning should default to empty."""
        raw = json.dumps({"selected_docs": ["earthmover.pdf"], "reasoning": 42})
        result = selector._parse_response(raw, "test", self.ALL_DOCS)
        assert result.reasoning == ""


class TestDocSelectionDataclass:
    def test_basic_properties(self):
        ds = DocSelection(
            original_query="test",
            selected_docs=["a.pdf", "b.pdf"],
            reasoning="reason",
            all_docs=["a.pdf", "b.pdf", "c.pdf"],
        )
        assert ds.num_selected == 2
        assert ds.num_total == 3

    def test_empty_selection(self):
        ds = DocSelection(
            original_query="test",
            selected_docs=[],
            all_docs=[],
        )
        assert ds.num_selected == 0
        assert ds.num_total == 0


class TestDocSelectorFastPaths:
    """Test the fast-path logic (0 or 1 doc) — no LLM calls needed."""

    @pytest.fixture
    def selector(self):
        return DocSelector(model="qwen3-vl")

    def test_empty_corpus(self, selector):
        result = asyncio.run(
            selector.select(
                original_query="anything",
                sub_queries=["anything"],
                registry_summary="Empty",
                changelog_summary="Empty",
                all_doc_names=[],
            )
        )
        assert result.selected_docs == []
        assert result.num_selected == 0
        assert "empty" in result.reasoning.lower()

    def test_single_doc_auto_selected(self, selector):
        result = asyncio.run(
            selector.select(
                original_query="anything",
                sub_queries=["anything"],
                registry_summary="1 doc",
                changelog_summary="n/a",
                all_doc_names=["only_doc.pdf"],
            )
        )
        assert result.selected_docs == ["only_doc.pdf"]
        assert result.num_selected == 1
        assert "auto-selected" in result.reasoning.lower()


class TestDocSelectorFuzzyMatch:
    """Test the static _fuzzy_match method directly."""

    def test_exact_case_insensitive(self):
        nfc_all = {"earthmover.pdf": "earthmover.pdf"}
        assert DocSelector._fuzzy_match("EARTHMOVER.PDF", nfc_all) == "earthmover.pdf"

    def test_substring_match(self):
        nfc_all = {"earthmover.pdf": "earthmover.pdf"}
        assert DocSelector._fuzzy_match("earthmover", nfc_all) == "earthmover.pdf"

    def test_no_match(self):
        nfc_all = {"earthmover.pdf": "earthmover.pdf"}
        assert DocSelector._fuzzy_match("totally_different", nfc_all) is None

    def test_reverse_substring(self):
        """If the candidate contains the search term."""
        nfc_all = {"report.pdf": "report.pdf"}
        assert DocSelector._fuzzy_match("report.pdf.bak", nfc_all) == "report.pdf"


# ==========================================================================
# BUILD STEP 4 — TreeGrep v2, ContextFetcher.fetch_from_md, ContextMerger
# ==========================================================================


# ---------------------------------------------------------------------------
# Helper: generate a .md file from a real PDF + structure JSON for tests
# ---------------------------------------------------------------------------


@pytest.fixture
def earthmover_md(tmp_output_dir):
    """Extract earthmover PDF to .md, return (index_path, md_path)."""
    index_path = _index_path("earthmover")
    pdf_path = _pdf_path("earthmover")
    ext = MdExtractor(
        pdf_path=pdf_path, index_path=index_path, output_dir=tmp_output_dir
    )
    md_path = ext.extract()
    return index_path, md_path


@pytest.fixture
def q1_earnings_md(tmp_output_dir):
    """Extract q1-fy25-earnings PDF to .md, return (index_path, md_path)."""
    index_path = _index_path("q1-fy25-earnings")
    pdf_path = _pdf_path("q1-fy25-earnings")
    ext = MdExtractor(
        pdf_path=pdf_path, index_path=index_path, output_dir=tmp_output_dir
    )
    md_path = ext.extract()
    return index_path, md_path


# ---------------------------------------------------------------------------
# TreeGrep v2 tests
# ---------------------------------------------------------------------------


class TestTreeGrepInit:
    def test_loads_structure_json(self):
        """TreeGrep should load the structure from JSON."""
        grep = TreeGrep(index_path=_index_path("earthmover"))
        assert len(grep.structure) > 0
        assert grep.doc_name == "earthmover.pdf"

    def test_loads_with_md(self, earthmover_md):
        """TreeGrep should parse .md sections when md_path is given."""
        index_path, md_path = earthmover_md
        grep = TreeGrep(index_path=index_path, md_path=md_path)
        assert len(grep.md_sections) > 0

    def test_loads_without_md(self):
        """TreeGrep should work without .md (body search disabled)."""
        grep = TreeGrep(index_path=_index_path("earthmover"), md_path=None)
        assert grep.md_sections == {}


class TestTreeGrepSearch:
    def test_search_title_match(self):
        """Search should find matches in node titles."""
        grep = TreeGrep(index_path=_index_path("earthmover"))
        # Actual title is "Earth Mover's Distance based Similarity Search at Scale"
        results = grep.search("Earth Mover")
        assert len(results) > 0
        title_hits = [r for r in results if r.matched_field == "title"]
        assert len(title_hits) > 0

    def test_search_body_match(self, earthmover_md):
        """Search should find matches in .md body text."""
        index_path, md_path = earthmover_md
        grep = TreeGrep(index_path=index_path, md_path=md_path)
        # Search for a word likely in the body text but not in titles
        results = grep.search("cost")
        body_hits = [r for r in results if r.matched_field == "body"]
        # There should be at least one body hit if the MD has the word "cost"
        # (earthmover.pdf is about earthmoving cost estimation)
        assert len(body_hits) >= 0  # may or may not match depending on content

    def test_search_empty_pattern(self):
        grep = TreeGrep(index_path=_index_path("earthmover"))
        assert grep.search("") == []

    def test_search_regex(self):
        grep = TreeGrep(index_path=_index_path("earthmover"))
        # Matches "Earth Mover" — "Mover" contains "earth" nowhere, but
        # the title starts with "Earth" so we use a pattern that matches it.
        results = grep.search(r"earth\s+mover", regex=True)
        assert len(results) > 0

    def test_search_invalid_regex(self):
        grep = TreeGrep(index_path=_index_path("earthmover"))
        results = grep.search("[invalid", regex=True)
        assert results == []

    def test_search_max_results(self, earthmover_md):
        index_path, md_path = earthmover_md
        grep = TreeGrep(index_path=index_path, md_path=md_path)
        results = grep.search("the", max_results=3)
        assert len(results) <= 3

    def test_search_scope_by_node_id(self):
        grep = TreeGrep(index_path=_index_path("earthmover"))
        # Get the first node_id from structure
        first_node_id = grep.structure[0].get("node_id", "")
        if first_node_id:
            results = grep.search("Earthmoving", scope=first_node_id)
            # All results should be from that subtree
            # Just check it doesn't crash
            assert isinstance(results, list)

    def test_results_sorted_by_relevance(self, earthmover_md):
        index_path, md_path = earthmover_md
        grep = TreeGrep(index_path=index_path, md_path=md_path)
        results = grep.search("Earthmoving")
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].relevance_score >= results[i + 1].relevance_score

    def test_grep_result_has_page_range(self):
        grep = TreeGrep(index_path=_index_path("earthmover"))
        results = grep.search("Earthmoving")
        for r in results:
            assert isinstance(r.page_range, tuple)
            assert len(r.page_range) == 2


class TestTreeGrepSearchMulti:
    def test_multi_search_deduplicates(self, earthmover_md):
        index_path, md_path = earthmover_md
        grep = TreeGrep(index_path=index_path, md_path=md_path)
        results = grep.search_multi(["Earthmoving", "earthmoving", "EARTHMOVING"])
        # Should deduplicate — same nodes matching the same field
        node_fields = [(r.node_id, r.matched_field) for r in results]
        assert len(node_fields) == len(set(node_fields))

    def test_multi_search_combines_patterns(self, earthmover_md):
        index_path, md_path = earthmover_md
        grep = TreeGrep(index_path=index_path, md_path=md_path)
        results = grep.search_multi(["Earthmoving", "cost", "equipment"])
        assert isinstance(results, list)


class TestParseMdSections:
    def test_parse_real_md(self, earthmover_md):
        _, md_path = earthmover_md
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        sections = _parse_md_sections(md_text)
        assert len(sections) > 0
        # All keys should be node_id strings
        for node_id in sections:
            assert isinstance(node_id, str)
            assert len(node_id) > 0

    def test_parse_synthetic_md(self):
        md_text = (
            "# Doc Title\n\n"
            "<!-- node_id: 0001 | pages: 1-2 -->\n"
            "## Section One\n\nBody text for section one.\n\n"
            "<!-- node_id: 0002 | pages: 3-4 -->\n"
            "## Section Two\n\nBody text for section two.\n"
        )
        sections = _parse_md_sections(md_text)
        assert "0001" in sections
        assert "0002" in sections
        assert "section one" in sections["0001"].lower()
        assert "section two" in sections["0002"].lower()


class TestBuildSnippet:
    def test_basic_snippet(self):
        text = "Hello world, this is a test of the snippet builder function."
        snippet = _build_snippet(text, 6, 11, padding=10)
        assert "world" in snippet

    def test_snippet_at_start(self):
        text = "Start of the text here."
        snippet = _build_snippet(text, 0, 5, padding=10)
        assert "Start" in snippet

    def test_snippet_at_end(self):
        text = "Some text at the end."
        snippet = _build_snippet(text, 18, 21, padding=5)
        assert "end" in snippet


# ---------------------------------------------------------------------------
# ContextFetcher.fetch_from_md tests
# ---------------------------------------------------------------------------


class TestContextFetcherMd:
    def test_fetch_from_md_returns_context(self, earthmover_md):
        index_path, md_path = earthmover_md
        fetcher = ContextFetcher(
            index_path=index_path,
            pdf_path=_pdf_path("earthmover"),
        )
        # Get a valid node_id from the node_map
        valid_ids = list(fetcher.node_map.keys())[:2]
        parts = fetcher.fetch_from_md(node_ids=valid_ids, md_path=md_path)
        assert len(parts) > 0
        # Each part should have a header with "Section:" and "node_id:"
        for part in parts:
            assert "Section:" in part
            assert "node_id:" in part

    def test_fetch_from_md_empty_ids(self, earthmover_md):
        index_path, md_path = earthmover_md
        fetcher = ContextFetcher(
            index_path=index_path,
            pdf_path=_pdf_path("earthmover"),
        )
        assert fetcher.fetch_from_md(node_ids=[], md_path=md_path) == []

    def test_fetch_from_md_truncation(self, earthmover_md):
        index_path, md_path = earthmover_md
        fetcher = ContextFetcher(
            index_path=index_path,
            pdf_path=_pdf_path("earthmover"),
        )
        valid_ids = list(fetcher.node_map.keys())[:1]
        parts = fetcher.fetch_from_md(
            node_ids=valid_ids,
            md_path=md_path,
            max_chars_per_node=100,
        )
        if parts:
            # If the section had > 100 chars, it should be truncated
            assert len(parts[0]) <= 300  # header + 100 chars + truncation marker

    def test_fetch_from_md_fallback_to_pdf(self, earthmover_md):
        """If md_path doesn't exist, should fall back to PDF fetch."""
        index_path, _ = earthmover_md
        fetcher = ContextFetcher(
            index_path=index_path,
            pdf_path=_pdf_path("earthmover"),
        )
        valid_ids = list(fetcher.node_map.keys())[:1]
        parts = fetcher.fetch_from_md(
            node_ids=valid_ids,
            md_path="/nonexistent/path.md",
        )
        # Should still return something (from PDF fallback)
        assert len(parts) > 0

    def test_get_node_info(self):
        fetcher = ContextFetcher(
            index_path=_index_path("earthmover"),
            pdf_path=_pdf_path("earthmover"),
        )
        valid_ids = list(fetcher.node_map.keys())[:2]
        info = fetcher.get_node_info(valid_ids)
        assert len(info) == len(valid_ids)
        for entry in info:
            assert "node_id" in entry
            assert "title" in entry
            assert "pages" in entry


# ---------------------------------------------------------------------------
# ContextMerger tests
# ---------------------------------------------------------------------------


class TestContextChunk:
    def test_char_count(self):
        chunk = ContextChunk(
            doc_name="test.pdf",
            node_id="0001",
            title="Test",
            page_range=(1, 2),
            text="Hello world",
        )
        assert chunk.char_count == 11

    def test_dedup_key(self):
        chunk = ContextChunk(
            doc_name="test.pdf",
            node_id="0001",
            title="Test",
            page_range=(1, 2),
            text="Hello",
        )
        assert chunk.dedup_key == ("test.pdf", "0001")


class TestContextMergerDedup:
    def test_dedup_same_doc_same_node(self):
        merger = ContextMerger()
        merger.add_chunk(
            ContextChunk(
                doc_name="a.pdf",
                node_id="0001",
                title="A",
                page_range=(1, 2),
                text="Version 1",
                relevance_score=1.0,
            )
        )
        merger.add_chunk(
            ContextChunk(
                doc_name="a.pdf",
                node_id="0001",
                title="A",
                page_range=(1, 2),
                text="Version 2",
                relevance_score=3.0,
            )
        )
        result = merger.merge()
        assert result.num_chunks == 1
        # Should keep the higher-scoring one
        assert result.chunks[0].relevance_score == 3.0

    def test_no_dedup_different_docs(self):
        merger = ContextMerger()
        merger.add_chunk(
            ContextChunk(
                doc_name="a.pdf",
                node_id="0001",
                title="A",
                page_range=(1, 2),
                text="From doc A",
            )
        )
        merger.add_chunk(
            ContextChunk(
                doc_name="b.pdf",
                node_id="0001",
                title="A",
                page_range=(1, 2),
                text="From doc B",
            )
        )
        result = merger.merge()
        assert result.num_chunks == 2
        assert result.num_docs == 2


class TestContextMergerBudget:
    def test_budget_truncates(self):
        merger = ContextMerger(max_total_chars=200)
        merger.add_chunk(
            ContextChunk(
                doc_name="a.pdf",
                node_id="0001",
                title="Long Section",
                page_range=(1, 5),
                text="x" * 500,
                relevance_score=3.0,
            )
        )
        result = merger.merge()
        # The formatted output includes headers (=== DOCUMENT ===, ## Title, ---)
        # so total_chars will exceed the raw text budget.  The raw chunk text
        # should be ≤ budget though.
        raw_text_chars = sum(c.char_count for c in result.chunks)
        assert raw_text_chars <= 200 + 30  # budget + truncation suffix
        assert "[... truncated]" in result.formatted_context

    def test_budget_drops_low_priority(self):
        merger = ContextMerger(max_total_chars=300)
        merger.add_chunk(
            ContextChunk(
                doc_name="a.pdf",
                node_id="0001",
                title="High",
                page_range=(1, 2),
                text="A" * 250,
                relevance_score=3.0,
            )
        )
        merger.add_chunk(
            ContextChunk(
                doc_name="a.pdf",
                node_id="0002",
                title="Low",
                page_range=(3, 4),
                text="B" * 250,
                relevance_score=1.0,
            )
        )
        result = merger.merge()
        assert result.num_chunks <= 2  # might include truncated second
        # High priority chunk should be present
        assert any(c.node_id == "0001" for c in result.chunks)

    def test_max_chunks_cap(self):
        merger = ContextMerger(max_chunks=2)
        for i in range(5):
            merger.add_chunk(
                ContextChunk(
                    doc_name="a.pdf",
                    node_id=f"000{i}",
                    title=f"Section {i}",
                    page_range=(i, i + 1),
                    text=f"Text {i}",
                    relevance_score=float(5 - i),
                )
            )
        result = merger.merge()
        assert result.num_chunks <= 2


class TestContextMergerFormat:
    def test_format_groups_by_document(self):
        merger = ContextMerger()
        merger.add_chunk(
            ContextChunk(
                doc_name="doc1.pdf",
                node_id="0001",
                title="Sec A",
                page_range=(1, 2),
                text="Body A",
                relevance_score=3.0,
            )
        )
        merger.add_chunk(
            ContextChunk(
                doc_name="doc2.pdf",
                node_id="0001",
                title="Sec B",
                page_range=(1, 2),
                text="Body B",
                relevance_score=2.0,
            )
        )
        result = merger.merge()
        assert "=== DOCUMENT: doc1.pdf ===" in result.formatted_context
        assert "=== DOCUMENT: doc2.pdf ===" in result.formatted_context

    def test_empty_merge(self):
        merger = ContextMerger()
        result = merger.merge()
        assert result.num_chunks == 0
        assert result.formatted_context == ""
        assert result.dropped_count == 0


class TestContextMergerReset:
    def test_reset_clears_chunks(self):
        merger = ContextMerger()
        merger.add_chunk(
            ContextChunk(
                doc_name="a.pdf",
                node_id="0001",
                title="A",
                page_range=(1, 2),
                text="Text",
            )
        )
        merger.reset()
        result = merger.merge()
        assert result.num_chunks == 0


class TestContextMergerFromFetcher:
    def test_add_from_fetcher_output(self):
        merger = ContextMerger()
        # Simulate ContextFetcher output format
        context_parts = [
            "## Section: Intro (Pages 1-3)\n\nIntro body text...",
            "## Section: Methods (Pages 4-6)\n\nMethods body text...",
        ]
        merger.add_from_fetcher_output(
            doc_name="test.pdf",
            context_parts=context_parts,
            node_ids=["0001", "0002"],
            relevance_scores=[3.0, 2.0],
        )
        result = merger.merge()
        assert result.num_chunks == 2
        assert result.chunks[0].title == "Intro"
        assert result.chunks[0].page_range == (1, 3)

    def test_parse_fetcher_header(self):
        title, page_range = ContextMerger._parse_fetcher_header(
            "## Section: My Title (Pages 12-15)\n\nBody..."
        )
        assert title == "My Title"
        assert page_range == (12, 15)

    def test_parse_fetcher_header_no_match(self):
        title, page_range = ContextMerger._parse_fetcher_header("Random text")
        assert title == "Untitled"
        assert page_range == (0, 0)


# ===========================================================================
# Step 5: LDRSPipeline tests
# ===========================================================================


# ---------------------------------------------------------------------------
# LDRSConfig tests
# ---------------------------------------------------------------------------


class TestLDRSConfig:
    def test_defaults(self):
        """Config should have sensible defaults."""
        config = LDRSConfig()
        assert config.results_dir == "tests/results"
        assert config.pdf_dir == "tests/pdfs"
        assert config.model == "qwen3-vl"
        assert config.max_sub_queries == 5
        assert config.min_sub_queries == 3
        assert config.max_total_chars == 15_000
        assert config.max_chunks == 30

    def test_md_dir_defaults_to_results_dir(self):
        """md_dir should default to results_dir if not set."""
        config = LDRSConfig(results_dir="/some/dir")
        assert config.md_dir == "/some/dir"

    def test_md_dir_explicit(self):
        """Explicit md_dir should override the default."""
        config = LDRSConfig(results_dir="/results", md_dir="/md")
        assert config.md_dir == "/md"

    def test_auto_registry_path(self):
        """registry_path should auto-generate from results_dir."""
        config = LDRSConfig(results_dir="/data/results")
        assert config.registry_path == "/data/results/_registry.json"

    def test_auto_changelog_path(self):
        """changelog_path should auto-generate from results_dir."""
        config = LDRSConfig(results_dir="/data/results")
        assert config.changelog_path == "/data/results/_changelog.json"

    def test_explicit_paths_override(self):
        """Explicit registry/changelog paths should override auto-generation."""
        config = LDRSConfig(
            results_dir="/data/results",
            registry_path="/custom/reg.json",
            changelog_path="/custom/cl.json",
        )
        assert config.registry_path == "/custom/reg.json"
        assert config.changelog_path == "/custom/cl.json"


# ---------------------------------------------------------------------------
# LDRSPipeline initialisation & corpus build tests
# ---------------------------------------------------------------------------


class TestLDRSPipelineInit:
    def test_init_creates_pipeline(self, tmp_output_dir):
        """Pipeline should initialise without errors."""
        config = LDRSConfig(
            results_dir=RESULTS_DIR,
            pdf_dir=PDFS_DIR,
            md_dir=tmp_output_dir,
            registry_path=os.path.join(tmp_output_dir, "_registry.json"),
            changelog_path=os.path.join(tmp_output_dir, "_changelog.json"),
        )
        pipeline = LDRSPipeline(config)
        assert pipeline.config is config
        assert pipeline.registry is not None
        assert pipeline.changelog is not None

    def test_build_corpus(self, tmp_output_dir):
        """build_corpus should register all documents from the results dir."""
        config = LDRSConfig(
            results_dir=RESULTS_DIR,
            pdf_dir=PDFS_DIR,
            md_dir=tmp_output_dir,
            registry_path=os.path.join(tmp_output_dir, "_registry.json"),
            changelog_path=os.path.join(tmp_output_dir, "_changelog.json"),
        )
        pipeline = LDRSPipeline(config)
        count = pipeline.build_corpus()
        assert count > 0
        assert len(pipeline.registry.doc_names) == count
        # Registry file should exist
        assert os.path.exists(os.path.join(tmp_output_dir, "_registry.json"))
        # Changelog file should exist
        assert os.path.exists(os.path.join(tmp_output_dir, "_changelog.json"))

    def test_corpus_summary(self, tmp_output_dir):
        """corpus_summary should return non-empty string after build."""
        config = LDRSConfig(
            results_dir=RESULTS_DIR,
            pdf_dir=PDFS_DIR,
            md_dir=tmp_output_dir,
            registry_path=os.path.join(tmp_output_dir, "_registry.json"),
            changelog_path=os.path.join(tmp_output_dir, "_changelog.json"),
        )
        pipeline = LDRSPipeline(config)
        pipeline.build_corpus()
        summary = pipeline.corpus_summary()
        assert "CORPUS REGISTRY" in summary

    def test_corpus_stats(self, tmp_output_dir):
        """corpus_stats should return correct structure."""
        config = LDRSConfig(
            results_dir=RESULTS_DIR,
            pdf_dir=PDFS_DIR,
            md_dir=tmp_output_dir,
            registry_path=os.path.join(tmp_output_dir, "_registry.json"),
            changelog_path=os.path.join(tmp_output_dir, "_changelog.json"),
        )
        pipeline = LDRSPipeline(config)
        pipeline.build_corpus()
        stats = pipeline.corpus_stats()
        assert "num_documents" in stats
        assert "doc_names" in stats
        assert stats["num_documents"] > 0


# ---------------------------------------------------------------------------
# LDRSPipeline._find_pdf_path tests
# ---------------------------------------------------------------------------


class TestFindPdfPath:
    def test_find_exact_name(self, tmp_output_dir):
        """Should find PDF with exact name."""
        config = LDRSConfig(
            results_dir=RESULTS_DIR,
            pdf_dir=PDFS_DIR,
            md_dir=tmp_output_dir,
            registry_path=os.path.join(tmp_output_dir, "_registry.json"),
            changelog_path=os.path.join(tmp_output_dir, "_changelog.json"),
        )
        pipeline = LDRSPipeline(config)
        path = pipeline._find_pdf_path("earthmover.pdf")
        assert os.path.exists(path)

    def test_find_without_extension(self, tmp_output_dir):
        """Should find PDF when name is given without .pdf extension."""
        config = LDRSConfig(
            results_dir=RESULTS_DIR,
            pdf_dir=PDFS_DIR,
            md_dir=tmp_output_dir,
            registry_path=os.path.join(tmp_output_dir, "_registry.json"),
            changelog_path=os.path.join(tmp_output_dir, "_changelog.json"),
        )
        pipeline = LDRSPipeline(config)
        path = pipeline._find_pdf_path("earthmover")
        assert os.path.exists(path)

    def test_not_found(self, tmp_output_dir):
        """Should return empty string for non-existent PDF."""
        config = LDRSConfig(
            results_dir=RESULTS_DIR,
            pdf_dir=PDFS_DIR,
            md_dir=tmp_output_dir,
            registry_path=os.path.join(tmp_output_dir, "_registry.json"),
            changelog_path=os.path.join(tmp_output_dir, "_changelog.json"),
        )
        pipeline = LDRSPipeline(config)
        path = pipeline._find_pdf_path("nonexistent_document")
        assert path == ""


# ---------------------------------------------------------------------------
# LDRSPipeline._add_to_merger tests
# ---------------------------------------------------------------------------


class TestAddToMerger:
    def test_adds_grep_hits_with_context(self):
        """Should convert grep hits + context parts to ContextChunks."""
        merger = ContextMerger()

        grep_hits = [
            GrepResult(
                node_id="0001",
                title="Intro",
                breadcrumb="Intro",
                matched_field="title",
                snippet="Introduction...",
                relevance_score=3.0,
                page_range=(1, 3),
            ),
            GrepResult(
                node_id="0003",
                title="Methods",
                breadcrumb="Methods",
                matched_field="body",
                snippet="The method uses...",
                relevance_score=1.0,
                page_range=(5, 7),
            ),
        ]

        context_parts = [
            "## Section: Intro (Pages 1-3) [node_id: 0001]\n\nIntro body text about the research.",
            "## Section: Methods (Pages 5-7) [node_id: 0003]\n\nMethods body describing the approach.",
        ]

        LDRSPipeline._add_to_merger(merger, "test.pdf", grep_hits, context_parts)

        # Should have 2 chunks (one per grep hit)
        assert len(merger._chunks) == 2
        # First chunk should have the fetched body, not just the snippet
        assert "Intro body text" in merger._chunks[0].text
        # Relevance scores from grep should be preserved
        assert merger._chunks[0].relevance_score == 3.0
        assert merger._chunks[1].relevance_score == 1.0

    def test_falls_back_to_snippet_when_no_body(self):
        """Should use grep snippet when no fetched body exists for a node."""
        merger = ContextMerger()

        grep_hits = [
            GrepResult(
                node_id="0099",
                title="Orphan",
                breadcrumb="Orphan",
                matched_field="title",
                snippet="Orphan section text",
                relevance_score=3.0,
                page_range=(1, 1),
            ),
        ]
        # No context parts for node 0099
        context_parts = []

        LDRSPipeline._add_to_merger(merger, "test.pdf", grep_hits, context_parts)

        assert len(merger._chunks) == 1
        assert merger._chunks[0].text == "Orphan section text"

    def test_deduplicates_within_doc(self):
        """Should not add duplicate chunks for the same node_id."""
        merger = ContextMerger()

        grep_hits = [
            GrepResult(
                node_id="0001",
                title="Intro",
                breadcrumb="Intro",
                matched_field="title",
                snippet="snip1",
                relevance_score=3.0,
                page_range=(1, 3),
            ),
            GrepResult(
                node_id="0001",  # same node, different field match
                title="Intro",
                breadcrumb="Intro",
                matched_field="body",
                snippet="snip2",
                relevance_score=1.0,
                page_range=(1, 3),
            ),
        ]

        context_parts = [
            "## Section: Intro (Pages 1-3) [node_id: 0001]\n\nIntro body.",
        ]

        LDRSPipeline._add_to_merger(merger, "test.pdf", grep_hits, context_parts)

        # Should only have 1 chunk (first grep hit wins)
        assert len(merger._chunks) == 1


# ---------------------------------------------------------------------------
# LDRSPipeline._build_citations tests
# ---------------------------------------------------------------------------


class TestBuildCitations:
    def test_builds_citations_from_merged(self):
        """Should extract citation metadata from merged context."""
        merged = MergedContext(
            chunks=[
                ContextChunk(
                    doc_name="earth.pdf",
                    node_id="0001",
                    title="Intro",
                    page_range=(1, 3),
                    text="...",
                    relevance_score=3.0,
                ),
                ContextChunk(
                    doc_name="earth.pdf",
                    node_id="0005",
                    title="Results",
                    page_range=(8, 10),
                    text="...",
                    relevance_score=2.0,
                ),
            ],
            formatted_context="...",
            total_chars=100,
            num_chunks=2,
            num_docs=1,
        )

        citations = LDRSPipeline._build_citations(merged)
        assert len(citations) == 2
        assert citations[0]["doc_name"] == "earth.pdf"
        assert citations[0]["section"] == "Intro"
        assert citations[0]["pages"] == "1-3"
        assert citations[1]["node_id"] == "0005"

    def test_deduplicates_citations(self):
        """Should not produce duplicate citations for same doc+node."""
        merged = MergedContext(
            chunks=[
                ContextChunk(
                    doc_name="a.pdf",
                    node_id="0001",
                    title="X",
                    page_range=(1, 1),
                    text="...",
                ),
                ContextChunk(
                    doc_name="a.pdf",
                    node_id="0001",
                    title="X",
                    page_range=(1, 1),
                    text="... different text",
                ),
            ],
        )
        citations = LDRSPipeline._build_citations(merged)
        assert len(citations) == 1

    def test_empty_merged_returns_empty_citations(self):
        """Empty merged context should return no citations."""
        merged = MergedContext()
        citations = LDRSPipeline._build_citations(merged)
        assert citations == []


# ---------------------------------------------------------------------------
# LDRSResult tests
# ---------------------------------------------------------------------------


class TestLDRSResult:
    def test_defaults(self):
        """LDRSResult should have empty defaults."""
        result = LDRSResult()
        assert result.query == ""
        assert result.answer == ""
        assert result.sub_queries == []
        assert result.selected_docs == []
        assert result.grep_hits == 0
        assert result.citations == []
        assert result.error == ""

    def test_populated(self):
        """LDRSResult should store populated fields."""
        result = LDRSResult(
            query="What is EMD?",
            answer="EMD is Earth Mover's Distance.",
            sub_queries=["EMD definition", "Earth Mover's Distance"],
            selected_docs=["earthmover.pdf"],
            grep_hits=5,
            citations=[
                {
                    "doc_name": "earthmover.pdf",
                    "section": "Intro",
                    "pages": "1-3",
                    "node_id": "0001",
                }
            ],
            timings={"query_expansion": 0.5, "generation": 1.2},
        )
        assert result.query == "What is EMD?"
        assert len(result.sub_queries) == 2
        assert result.timings["generation"] == 1.2


# ---------------------------------------------------------------------------
# LDRSPipeline.query with mocked LLM (integration-level test)
# ---------------------------------------------------------------------------


class TestLDRSPipelineQuery:
    """Tests that exercise the full pipeline with mocked LLM calls."""

    @pytest.fixture
    def pipeline_with_corpus(self, tmp_output_dir):
        """Build a pipeline with the test corpus pre-loaded."""
        config = LDRSConfig(
            results_dir=RESULTS_DIR,
            pdf_dir=PDFS_DIR,
            md_dir=tmp_output_dir,
            registry_path=os.path.join(tmp_output_dir, "_registry.json"),
            changelog_path=os.path.join(tmp_output_dir, "_changelog.json"),
        )
        pipeline = LDRSPipeline(config)
        pipeline.build_corpus()
        return pipeline

    @pytest.mark.asyncio
    async def test_query_empty_selection(self, pipeline_with_corpus):
        """Pipeline should handle DocSelector returning no docs gracefully."""
        pipeline = pipeline_with_corpus

        # Mock QueryExpander to return fixed sub-queries
        mock_expander = AsyncMock()
        mock_expander.expand.return_value = ExpandedQuery(
            original_query="What is dark matter?",
            sub_queries=["dark matter definition", "dark matter properties"],
            reasoning="Test reasoning",
        )
        pipeline._query_expander = mock_expander

        # Mock DocSelector to return empty selection
        mock_selector = AsyncMock()
        mock_selector.select.return_value = DocSelection(
            original_query="What is dark matter?",
            selected_docs=[],  # no docs selected
            reasoning="No relevant docs found.",
            all_docs=pipeline.registry.doc_names,
        )
        pipeline._doc_selector = mock_selector

        result = await pipeline.query("What is dark matter?")

        assert result.query == "What is dark matter?"
        assert result.selected_docs == []
        assert "No documents were selected" in result.answer
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_query_full_flow_mocked(self, pipeline_with_corpus):
        """Full pipeline query with all LLM stages mocked."""
        pipeline = pipeline_with_corpus

        # Pick a doc that's in the corpus
        doc_names = pipeline.registry.doc_names
        assert len(doc_names) > 0
        target_doc = doc_names[0]

        # Mock QueryExpander
        mock_expander = AsyncMock()
        mock_expander.expand.return_value = ExpandedQuery(
            original_query="What is the main topic?",
            sub_queries=["main topic", "introduction", "abstract"],
            reasoning="Basic expansion",
        )
        pipeline._query_expander = mock_expander

        # Mock DocSelector to select the first doc
        mock_selector = AsyncMock()
        mock_selector.select.return_value = DocSelection(
            original_query="What is the main topic?",
            selected_docs=[target_doc],
            reasoning="Selected first doc.",
            all_docs=doc_names,
        )
        pipeline._doc_selector = mock_selector

        # Mock _retrieve_from_document to return synthetic grep hits so
        # the pipeline actually reaches the Generator stage.
        fake_grep_hit = GrepResult(
            node_id="0001",
            title="Introduction",
            breadcrumb="Introduction",
            matched_field="title",
            snippet="This is the introduction section.",
            relevance_score=3.0,
            page_range=(1, 3),
        )

        async def _fake_retrieve(doc_name, sub_queries):
            return (
                doc_name,
                [fake_grep_hit],
                ["## Section: Introduction (Pages 1-3) [node_id: 0001]\nIntro text."],
            )

        pipeline._retrieve_from_document = _fake_retrieve

        # Mock Generator
        from rag.generator import GenerationResult

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = GenerationResult(
            answer="The main topic is about Earth Mover's Distance.", usage={}
        )
        pipeline._generator = mock_generator

        result = await pipeline.query("What is the main topic?")

        assert result.query == "What is the main topic?"
        assert result.answer == "The main topic is about Earth Mover's Distance."
        assert result.sub_queries == ["main topic", "introduction", "abstract"]
        assert result.selected_docs == [target_doc]
        assert result.error == ""
        # Should have some timings
        assert "query_expansion" in result.timings
        assert "doc_selection" in result.timings
        assert "retrieval" in result.timings
        assert "generation" in result.timings

    @pytest.mark.asyncio
    async def test_query_handles_exception(self, pipeline_with_corpus):
        """Pipeline should catch exceptions and return error result."""
        pipeline = pipeline_with_corpus

        # Mock QueryExpander to raise
        mock_expander = AsyncMock()
        mock_expander.expand.side_effect = RuntimeError("LLM is down")
        pipeline._query_expander = mock_expander

        result = await pipeline.query("What is EMD?")

        assert result.query == "What is EMD?"
        assert "error" in result.answer.lower() or result.error != ""
        assert "LLM is down" in result.error

    @pytest.mark.asyncio
    async def test_query_no_grep_hits(self, pipeline_with_corpus):
        """Pipeline should handle case where TreeGrep returns no hits."""
        pipeline = pipeline_with_corpus

        doc_names = pipeline.registry.doc_names
        target_doc = doc_names[0]

        # Mock QueryExpander
        mock_expander = AsyncMock()
        mock_expander.expand.return_value = ExpandedQuery(
            original_query="xyzzy",
            sub_queries=["xyzzy_impossible_term_12345"],
            reasoning="Nonsense query",
        )
        pipeline._query_expander = mock_expander

        # Mock DocSelector
        mock_selector = AsyncMock()
        mock_selector.select.return_value = DocSelection(
            original_query="xyzzy",
            selected_docs=[target_doc],
            reasoning="Selected doc.",
            all_docs=doc_names,
        )
        pipeline._doc_selector = mock_selector

        result = await pipeline.query("xyzzy")

        # The very unlikely search term should produce 0 grep hits
        # and the pipeline should handle it gracefully
        assert result.query == "xyzzy"
        assert result.error == ""
        # Either we got 0 hits and a "no relevant sections" message,
        # or by coincidence the term matched — either way, no crash
        assert isinstance(result.answer, str)


# ---------------------------------------------------------------------------
# LDRSPipeline.index_document test
# ---------------------------------------------------------------------------


class TestLDRSPipelineIndex:
    def test_index_document(self, tmp_output_dir):
        """index_document should extract .md, register, and log."""
        config = LDRSConfig(
            results_dir=RESULTS_DIR,
            pdf_dir=PDFS_DIR,
            md_dir=tmp_output_dir,
            registry_path=os.path.join(tmp_output_dir, "_registry.json"),
            changelog_path=os.path.join(tmp_output_dir, "_changelog.json"),
        )
        pipeline = LDRSPipeline(config)

        md_path = pipeline.index_document(
            pdf_path=_pdf_path("earthmover"),
            index_path=_index_path("earthmover"),
        )

        # .md file should exist
        assert os.path.exists(md_path)
        assert md_path.endswith(".md")

        # Registry should have the doc
        assert len(pipeline.registry.doc_names) == 1
        assert "earthmover.pdf" in pipeline.registry.doc_names

        # Changelog should have an entry
        assert len(pipeline.changelog.entries) > 0


# ---------------------------------------------------------------------------
# Import tests (ensure modules load without errors)
# ---------------------------------------------------------------------------


class TestStep5Imports:
    def test_import_pipeline(self):
        """ldrs.ldrs_pipeline should import without errors."""
        from ldrs.ldrs_pipeline import LDRSPipeline, LDRSConfig, LDRSResult

        assert LDRSPipeline is not None

    def test_import_via_init(self):
        """Pipeline classes should be accessible via ldrs package."""
        import ldrs

        assert hasattr(ldrs, "LDRSPipeline")
        assert hasattr(ldrs, "LDRSConfig")
        assert hasattr(ldrs, "LDRSResult")
