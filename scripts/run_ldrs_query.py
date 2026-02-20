#!/usr/bin/env python3
"""
LDRS v2 CLI — Interactive query runner.

Usage:
    # Single query
    python scripts/run_ldrs_query.py --query "What is Earth Mover's Distance?"

    # Interactive mode (prompts for queries)
    python scripts/run_ldrs_query.py

    # Custom directories
    python scripts/run_ldrs_query.py \\
        --results-dir tests/results \\
        --pdf-dir tests/pdfs \\
        --query "Explain the EMD algorithm"

    # Verbose logging
    python scripts/run_ldrs_query.py --verbose --query "inflation rate"

Environment:
    Requires API_KEY and BASE_URL in .env (loaded automatically).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ldrs.ldrs_pipeline import LDRSConfig, LDRSPipeline, LDRSResult


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _print_result(result: LDRSResult) -> None:
    """Pretty-print an LDRSResult to the terminal."""
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"  QUERY: {result.query}")
    print(sep)

    if result.error:
        print(f"\n  ERROR: {result.error}\n")

    print(f"\n  ANSWER:\n")
    # Indent answer for readability
    for line in result.answer.split("\n"):
        print(f"    {line}")

    print(f"\n{'-' * 72}")
    print(f"  Sub-queries ({len(result.sub_queries)}):")
    for i, sq in enumerate(result.sub_queries, 1):
        print(f"    {i}. {sq}")

    print(f"\n  Selected docs ({len(result.selected_docs)}):")
    for doc in result.selected_docs:
        print(f"    - {doc}")

    print(f"\n  Grep hits: {result.grep_hits}")

    if result.merged_context:
        mc = result.merged_context
        print(
            f"  Merged context: {mc.num_chunks} chunks, {mc.total_chars} chars, {mc.num_docs} docs"
        )
        if mc.dropped_count > 0:
            print(f"  Dropped chunks: {mc.dropped_count}")

    if result.citations:
        print(f"\n  Citations ({len(result.citations)}):")
        for c in result.citations:
            print(
                f"    [{c['node_id']}] {c['section']} (Pages {c['pages']}) — {c['doc_name']}"
            )

    print(f"\n  Timings:")
    total = 0.0
    for stage, elapsed in result.timings.items():
        print(f"    {stage:20s}  {elapsed:.2f}s")
        total += elapsed
    print(f"    {'TOTAL':20s}  {total:.2f}s")

    if result.expansion_reasoning:
        print(f"\n  Expansion reasoning: {result.expansion_reasoning[:200]}")
    if result.selection_reasoning:
        print(f"  Selection reasoning: {result.selection_reasoning[:200]}")

    print(f"\n{sep}\n")


async def _run_single_query(pipeline: LDRSPipeline, query: str) -> LDRSResult:
    """Run a single query and return the result."""
    print(f"\nProcessing query: {query!r}")
    print(
        "This may take a moment (LLM calls for expansion + selection + generation)...\n"
    )

    t0 = time.monotonic()
    result = await pipeline.query(query)
    elapsed = time.monotonic() - t0

    print(f"Query completed in {elapsed:.2f}s")
    return result


async def _interactive_mode(pipeline: LDRSPipeline) -> None:
    """Run an interactive query loop."""
    print("\n" + "=" * 72)
    print("  LDRS v2 — Interactive Query Mode")
    print("  Type your question, or 'quit' / 'exit' to stop.")
    print("  Type 'corpus' to see the corpus summary.")
    print("  Type 'stats' to see corpus statistics.")
    print("=" * 72 + "\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if query.lower() == "corpus":
            print(pipeline.corpus_summary())
            continue

        if query.lower() == "stats":
            stats = pipeline.corpus_stats()
            print(json.dumps(stats, indent=2))
            continue

        result = await _run_single_query(pipeline, query)
        _print_result(result)


async def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LDRS v2 — Query your document corpus with AI-powered retrieval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Single query to run (omit for interactive mode).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="tests/results",
        help="Directory containing *_structure.json files (default: tests/results).",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="tests/pdfs",
        help="Directory containing PDF files (default: tests/pdfs).",
    )
    parser.add_argument(
        "--md-dir",
        type=str,
        default=None,
        help="Directory for cached .md files (default: same as results-dir).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl",
        help="LLM model name (default: qwen3-vl).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output result as JSON instead of pretty-printed text.",
    )

    args = parser.parse_args()
    _setup_logging(verbose=args.verbose)

    # Build config
    config = LDRSConfig(
        results_dir=args.results_dir,
        pdf_dir=args.pdf_dir,
        md_dir=args.md_dir,
        model=args.model,
    )

    # Initialize pipeline
    print("Initialising LDRS v2 pipeline...")
    pipeline = LDRSPipeline(config)

    print(f"Building corpus from {config.results_dir} ...")
    count = pipeline.build_corpus()
    print(f"Corpus ready: {count} documents registered.\n")

    if args.query:
        # Single query mode
        result = await _run_single_query(pipeline, args.query)

        if args.json_output:
            # Serialise to JSON
            output = {
                "query": result.query,
                "answer": result.answer,
                "sub_queries": result.sub_queries,
                "selected_docs": result.selected_docs,
                "grep_hits": result.grep_hits,
                "citations": result.citations,
                "timings": result.timings,
                "error": result.error,
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            _print_result(result)
    else:
        # Interactive mode
        await _interactive_mode(pipeline)


if __name__ == "__main__":
    asyncio.run(main())
